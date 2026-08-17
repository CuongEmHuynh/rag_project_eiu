"""Orchestrator SAHC-v2: OCR JSON -> document tree -> atomic units -> parent/child chunks."""

from __future__ import annotations

import logging
import re
import statistics
import uuid
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from .models import (
    AtomicUnit,
    Chunk,
    ChunkingConfig,
    DocumentNode,
    DocumentTree,
    OCRDocument,
    PackedUnit,
    ParsedTable,
)
from .normalize import is_indexable_block, sort_blocks_in_reading_order
from .ocr_parser import group_blocks_by_page, load_ocr_json
from .retrieval_text import build_retrieval_text
from .structure_parser import build_document_tree, find_ancestor, iter_nodes
from .table_parser import (
    parse_document_tables,
    reconstruct_cross_page_tables,
    serialize_table_row,
)
from .token_counter import TokenCounter
from .token_packer import pack_atomic_units
from .validators import validate_chunks


LOGGER = logging.getLogger(__name__)
_PRIMARY_PARENT_TYPES = {
    "article", "preamble", "recipients", "signature", "metadata", "decision_heading", "document"
}


def build_document_chunks(
    json_path: str,
    document_meta: dict[str, Any],
    embedding_model: Any,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Entry point production đọc OCR JSON và trả parent + embedding child chunks."""

    ocr_document = load_ocr_json(json_path)
    counter = TokenCounter(embedding_model)
    return build_chunks_from_ocr_document(
        ocr_document,
        document_meta=document_meta,
        token_counter=counter,
        config=config,
    )


def build_document_chunks_v2(
    json_path: str,
    document_meta: dict[str, Any],
    embedding_model: Any,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Alias explicit version để tích hợp feature flag mà không phá tên API chung."""

    return build_document_chunks(json_path, document_meta, embedding_model, config)


def build_chunks_from_ocr_document(
    ocr_document: OCRDocument,
    document_meta: dict[str, Any],
    token_counter: TokenCounter,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Pipeline thuần từ OCRDocument, thuận tiện cho unit test và service integration."""

    policy = config or ChunkingConfig()
    document_id = _document_id(document_meta, ocr_document)
    indexable_blocks = [block for block in ocr_document.blocks if is_indexable_block(block)]
    ordered_blocks = sort_blocks_in_reading_order(indexable_blocks)
    tree = build_document_tree(
        ordered_blocks,
        document_id=document_id,
        document_title=_meta_value(document_meta, "Summary", "summary") or None,
    )

    physical_tables = parse_document_tables(ordered_blocks, tree)
    if policy.merge_cross_page_tables:
        logical_tables = reconstruct_cross_page_tables(
            physical_tables,
            group_blocks_by_page(ordered_blocks),
            threshold=policy.table_continuation_threshold,
        )
    else:
        logical_tables = physical_tables

    atomic_units = create_atomic_units(tree, logical_tables, document_meta, policy)
    atomic_units.sort(key=_atomic_sort_key)
    parent_chunks, parent_id_map, table_parent_id_map = _build_parent_chunks(
        tree, logical_tables, atomic_units, document_id, document_meta, token_counter
    )

    def candidate_builder(units: list[AtomicUnit]) -> str:
        """Dựng retrieval text tạm để packer đếm cả contextual prefix."""

        draft = _draft_chunk_from_units(units, document_id)
        return build_retrieval_text(document_meta, draft)

    packed_units = pack_atomic_units(
        atomic_units,
        token_counter,
        policy,
        candidate_text_builder=candidate_builder,
    )
    child_chunks = _build_child_chunks(
        packed_units,
        document_id,
        document_meta,
        token_counter,
        parent_id_map,
        table_parent_id_map,
        tree,
    )
    _link_parent_children(parent_chunks, child_chunks)
    chunks = [*parent_chunks, *child_chunks]
    validate_chunks(chunks, token_counter, policy)
    _log_document_stats(
        document_id,
        ocr_document,
        ordered_blocks,
        tree,
        logical_tables,
        parent_chunks,
        child_chunks,
    )
    return chunks


def create_atomic_units(
    tree: DocumentTree,
    tables: Iterable[ParsedTable],
    document_meta: dict[str, Any],
    config: ChunkingConfig | None = None,
) -> list[AtomicUnit]:
    """Chuyển tree prose và logical table rows thành semantic atomic units."""

    policy = config or ChunkingConfig()
    units: list[AtomicUnit] = []
    for node in iter_nodes(tree):
        if node.node_type in {"document", "table", "preamble", "metadata"}:
            continue
        if node.node_type == "recipients" and not policy.index_recipients:
            continue
        if node.node_type == "recipient_item":
            recipient_parent = find_ancestor(tree, node.node_id, {"recipients"})
            if not policy.index_recipients or recipient_parent is None:
                continue
        if node.node_type == "signature" and not policy.index_signature:
            continue

        parent = _primary_parent(tree, node)
        if parent is None or not node.text_normalized.strip():
            continue
        unit_type = _unit_type_for_node(node)
        units.append(
            AtomicUnit(
                unit_id=f"unit:{node.node_id}",
                unit_type=unit_type,
                parent_id=parent.node_id,
                section_path=list(node.section_path or parent.section_path),
                raw_text=node.text_raw,
                normalized_text=node.text_normalized,
                page_start=node.page_start,
                page_end=node.page_end,
                source_block_ids=list(node.source_block_ids),
                metadata={
                    "source_node_id": node.node_id,
                    "source_node_type": node.node_type,
                    "sort_order": _source_order(node.source_block_ids),
                },
            )
        )

    for table in tables:
        parent_id = str(table.metadata.get("structural_parent_id") or tree.root_id)
        row_sources = table.metadata.get("row_source_block_ids", [])
        section_path = list(table.metadata.get("section_path") or ["Bảng"])
        for row_index, row in enumerate(table.rows):
            body = serialize_table_row(
                table,
                row,
                document_meta,
                section_path,
                include_context=False,
            )
            if not body.strip():
                continue
            source_ids = (
                list(row_sources[row_index])
                if row_index < len(row_sources) and row_sources[row_index]
                else list(table.source_block_ids)
            )
            row_page = _page_from_sources(source_ids, table.page_start)
            units.append(
                AtomicUnit(
                    unit_id=f"unit:{table.table_id}:row:{row_index}",
                    unit_type="table_row",
                    parent_id=parent_id,
                    section_path=section_path,
                    raw_text=" | ".join(row),
                    normalized_text=body,
                    page_start=row_page,
                    page_end=row_page,
                    source_block_ids=source_ids,
                    metadata={
                        "table_id": table.table_id,
                        "table_row_index": row_index,
                        "table_schema": list(table.headers),
                        "table_schema_type": table.metadata.get("schema_type", "generic"),
                        "cross_page_table": bool(table.metadata.get("cross_page")),
                        "sort_order": _source_order(source_ids, row_index),
                    },
                )
            )
    return units


def _build_parent_chunks(
    tree: DocumentTree,
    tables: list[ParsedTable],
    units: list[AtomicUnit],
    document_id: str,
    document_meta: dict[str, Any],
    counter: TokenCounter,
) -> tuple[list[Chunk], dict[str, str], dict[str, str]]:
    """Tạo primary structural parents và secondary logical table parents."""

    units_by_parent: dict[str, list[AtomicUnit]] = {}
    for unit in units:
        units_by_parent.setdefault(unit.parent_id, []).append(unit)

    parent_chunks: list[Chunk] = []
    parent_id_map: dict[str, str] = {}
    for parent_index, logical_parent_id in enumerate(units_by_parent):
        node = tree.nodes.get(logical_parent_id, tree.root)
        parent_uuid = _stable_uuid(
            document_id,
            f"v2:parent:{node.node_type}:{logical_parent_id}:{','.join(node.source_block_ids)}",
        )
        parent_id_map[logical_parent_id] = parent_uuid
        grouped = units_by_parent[logical_parent_id]
        raw_text = "\n\n".join(unit.raw_text for unit in grouped if unit.raw_text)
        normalized_text = "\n\n".join(unit.normalized_text for unit in grouped if unit.normalized_text)
        source_ids = _unique_sources(grouped)
        draft = Chunk(
            chunk_id=parent_uuid,
            document_id=document_id,
            parent_id=None,
            chunk_index=parent_index,
            chunk_type=node.node_type,
            section_path=list(node.section_path),
            page_start=min(unit.page_start for unit in grouped),
            page_end=max(unit.page_end for unit in grouped),
            raw_text=raw_text,
            normalized_text=normalized_text,
            retrieval_text="",
            token_count=0,
            source_block_ids=source_ids,
            metadata={
                "record_type": "parent",
                "is_embedding_child": False,
                "parent_type": node.node_type,
                "logical_node_id": logical_parent_id,
                "child_ids": [],
                "chunking_version": "v2",
            },
        )
        draft.retrieval_text = build_retrieval_text(document_meta, draft)
        draft.token_count = counter.count(draft.retrieval_text)
        parent_chunks.append(draft)

    table_parent_id_map: dict[str, str] = {}
    for table in tables:
        table_units = [unit for unit in units if unit.metadata.get("table_id") == table.table_id]
        if not table_units:
            continue
        table_parent_uuid = _stable_uuid(
            document_id,
            f"v2:parent:table:{table.table_id}:{','.join(table.source_block_ids)}",
        )
        table_parent_id_map[table.table_id] = table_parent_uuid
        primary_parent_uuid = parent_id_map.get(
            str(table.metadata.get("structural_parent_id") or tree.root_id)
        )
        draft = Chunk(
            chunk_id=table_parent_uuid,
            document_id=document_id,
            parent_id=primary_parent_uuid,
            chunk_index=len(parent_chunks),
            chunk_type="table",
            section_path=list(table.metadata.get("section_path") or ["Bảng"]),
            page_start=table.page_start,
            page_end=table.page_end,
            raw_text="\n".join(unit.raw_text for unit in table_units),
            normalized_text="\n\n".join(unit.normalized_text for unit in table_units),
            retrieval_text="",
            token_count=0,
            source_block_ids=list(table.source_block_ids),
            metadata={
                "record_type": "parent",
                "is_embedding_child": False,
                "parent_type": "table",
                "table_id": table.table_id,
                "table_schema": list(table.headers),
                "cross_page_table": bool(table.metadata.get("cross_page")),
                "child_ids": [],
                "chunking_version": "v2",
            },
        )
        draft.retrieval_text = build_retrieval_text(document_meta, draft)
        draft.token_count = counter.count(draft.retrieval_text)
        parent_chunks.append(draft)
    return parent_chunks, parent_id_map, table_parent_id_map


def _build_child_chunks(
    packed_units: list[PackedUnit],
    document_id: str,
    document_meta: dict[str, Any],
    counter: TokenCounter,
    parent_id_map: dict[str, str],
    table_parent_id_map: dict[str, str],
    tree: DocumentTree,
) -> list[Chunk]:
    """Materialize PackedUnit thành deterministic child Chunk và contextual text."""

    children: list[Chunk] = []
    for child_index, packed in enumerate(packed_units):
        first = packed.units[0]
        logical_parent = first.parent_id
        parent_uuid = parent_id_map[logical_parent]
        chunk_type = first.unit_type
        signature = (
            f"v2:child:{chunk_type}:{' > '.join(first.section_path)}:"
            f"{','.join(packed.source_block_ids)}:{','.join(packed.metadata['atomic_unit_ids'])}"
        )
        metadata = {key: value for key, value in packed.metadata.items() if key != "sort_order"}
        metadata.update(
            {
                "record_type": "child",
                "is_embedding_child": True,
                "chunking_version": "v2",
                "parent_type": tree.nodes.get(logical_parent, tree.root).node_type,
            }
        )
        table_id = metadata.get("table_id")
        if table_id:
            metadata["table_parent_id"] = table_parent_id_map.get(str(table_id))
        child = Chunk(
            chunk_id=_stable_uuid(document_id, signature),
            document_id=document_id,
            parent_id=parent_uuid,
            chunk_index=child_index,
            chunk_type=chunk_type,
            section_path=list(first.section_path),
            page_start=packed.page_start,
            page_end=packed.page_end,
            raw_text=packed.raw_text,
            normalized_text=packed.normalized_text,
            retrieval_text="",
            token_count=0,
            source_block_ids=list(packed.source_block_ids),
            metadata=metadata,
        )
        child.retrieval_text = build_retrieval_text(document_meta, child)
        child.token_count = counter.count(child.retrieval_text)
        children.append(child)
    return children


def _draft_chunk_from_units(units: list[AtomicUnit], document_id: str) -> Chunk:
    """Tạo Chunk không ID để reuse retrieval text builder trong prefix-aware packing."""

    first = units[0]
    return Chunk(
        chunk_id="draft",
        document_id=document_id,
        parent_id=first.parent_id,
        chunk_index=-1,
        chunk_type=first.unit_type,
        section_path=list(first.section_path),
        page_start=min(unit.page_start for unit in units),
        page_end=max(unit.page_end for unit in units),
        raw_text="\n\n".join(unit.raw_text for unit in units),
        normalized_text="\n\n".join(unit.normalized_text for unit in units),
        retrieval_text="",
        token_count=0,
        source_block_ids=_unique_sources(units),
        metadata=dict(first.metadata),
    )


def _primary_parent(tree: DocumentTree, node: DocumentNode) -> DocumentNode | None:
    """Chọn parent context rộng: Điều/preamble/recipients/signature thay vì Khoản/Điểm."""

    if node.node_type in _PRIMARY_PARENT_TYPES:
        return node
    return find_ancestor(tree, node.node_id, _PRIMARY_PARENT_TYPES)


def _unit_type_for_node(node: DocumentNode) -> str:
    """Map structural node type sang vocabulary AtomicUnit/child chunk."""

    mapping = {
        "article": "article_intro",
        "decision_heading": "decision_heading",
        "recipients": "recipient_item",
        "signature": "signature",
    }
    return mapping.get(node.node_type, node.node_type)


def _link_parent_children(parents: list[Chunk], children: list[Chunk]) -> None:
    """Ghi child_ids vào primary article parent và secondary table parent."""

    by_id = {parent.chunk_id: parent for parent in parents}
    for child in children:
        parent = by_id.get(child.parent_id or "")
        if parent:
            parent.metadata.setdefault("child_ids", []).append(child.chunk_id)
        table_parent = by_id.get(str(child.metadata.get("table_parent_id") or ""))
        if table_parent:
            table_parent.metadata.setdefault("child_ids", []).append(child.chunk_id)


def _document_id(document_meta: dict[str, Any], document: OCRDocument) -> str:
    """Resolve document ID từ metadata; fallback input_file nhưng không dùng ID mẫu hard-code."""

    value = _meta_value(document_meta, "Id", "id", "document_id") or document.input_file
    if not value:
        raise ValueError("document_meta cần Id/document_id hoặc OCR JSON cần input_file")
    return str(value)


def _meta_value(meta: dict[str, Any], *keys: str) -> str:
    """Lấy metadata đầu tiên có giá trị và loại ``None``/whitespace."""

    for key in keys:
        value = meta.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _stable_uuid(document_id: str, name: str) -> str:
    """Sinh UUID5 deterministic với namespace document, hỗ trợ ID không phải UUID."""

    try:
        namespace = uuid.UUID(str(document_id))
    except (ValueError, TypeError, AttributeError):
        namespace = uuid.uuid5(uuid.NAMESPACE_URL, str(document_id))
    return str(uuid.uuid5(namespace, name))


def _unique_sources(units: Iterable[AtomicUnit]) -> list[str]:
    """Deduplicate source block IDs mà vẫn giữ thứ tự xuất hiện."""

    result: list[str] = []
    for unit in units:
        for source_id in unit.source_block_ids:
            if source_id not in result:
                result.append(source_id)
    return result


def _source_order(source_ids: list[str], sub_index: int = 0) -> tuple[int, int, int]:
    """Trích page/block từ stable source ID để interleave prose và table rows."""

    if not source_ids:
        return (10**9, 10**9, sub_index)
    match = re.search(r"page_(\d+)_block_(\d+)", source_ids[0])
    if not match:
        return (10**9, 10**9, sub_index)
    return (int(match.group(1)), int(match.group(2)), sub_index)


def _atomic_sort_key(unit: AtomicUnit) -> tuple[int, int, int, str]:
    """Sort atomic units deterministic theo provenance rồi unit ID."""

    order = unit.metadata.get("sort_order") or _source_order(unit.source_block_ids)
    return (int(order[0]), int(order[1]), int(order[2]), unit.unit_id)


def _page_from_sources(source_ids: list[str], fallback: int) -> int:
    """Suy ra page thật của table row từ source block provenance."""

    if source_ids:
        match = re.search(r"page_(\d+)_block_", source_ids[0])
        if match:
            return int(match.group(1))
    return fallback


def _log_document_stats(
    document_id: str,
    document: OCRDocument,
    blocks: list,
    tree: DocumentTree,
    tables: list[ParsedTable],
    parents: list[Chunk],
    children: list[Chunk],
) -> None:
    """Log metrics bắt buộc mỗi document để audit chất lượng chunking."""

    token_counts = [child.token_count for child in children]
    LOGGER.info(
        "[chunking-v2] doc=%s pages=%d blocks=%d noise_blocks=%d articles=%d "
        "legal_basis=%d tables=%d cross_page_tables=%d parents=%d children=%d "
        "max_tokens=%d avg_tokens=%.1f fallback_token_splits=%d",
        document_id,
        document.page_count,
        len(blocks),
        len(document.blocks) - len(blocks),
        sum(1 for _ in iter_nodes(tree, "article")),
        sum(1 for _ in iter_nodes(tree, "legal_basis")),
        len(tables),
        sum(bool(table.metadata.get("cross_page")) for table in tables),
        len(parents),
        len(children),
        max(token_counts, default=0),
        statistics.fmean(token_counts) if token_counts else 0.0,
        sum(child.metadata.get("split_fallback") == "token_window" for child in children),
    )
