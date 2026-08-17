"""Deterministic parser xây cây cấu trúc văn bản hành chính Việt Nam."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable, Iterator

from .models import DocumentNode, DocumentTree, OCRBlock
from .normalize import is_indexable_block, sort_blocks_in_reading_order


ARTICLE_RE = re.compile(r"^\s*Điều\s+(\d+[A-Za-z]?)\s*[:.\-]?\s*", re.IGNORECASE)
CLAUSE_RE = re.compile(r"^\s*(\d{1,3})[.)]\s+", re.IGNORECASE)
POINT_RE = re.compile(r"^\s*([a-zđ])[.)]\s+", re.IGNORECASE)
LEGAL_BASIS_RE = re.compile(
    r"^\s*[-–—]?\s*(?:Căn\s+c[ứửưúủu](?:\s+vào)?|Theo\s+đề\s+nghị)\b",
    re.IGNORECASE,
)
RECIPIENT_RE = re.compile(r"^\s*Nơi\s+nhận\s*[:.]?", re.IGNORECASE)
DECISION_RE = re.compile(r"^\s*QUY[ẾE]T\s+ĐỊNH\b", re.IGNORECASE)
SIGNATURE_RE = re.compile(
    r"^\s*(?:KT\.?\s+|TL\.?\s+|TUQ\.?\s+)?(?:HIỆU TRƯỞNG|GIÁM ĐỐC|"
    r"PHÓ\s+(?:HIỆU TRƯỞNG|GIÁM ĐỐC)|CHỦ TỊCH|THỦ TRƯỞNG|BỘ TRƯỞNG|"
    r"NGƯỜI KÝ|ĐÃ KÝ)\b",
    re.IGNORECASE,
)


def detect_article_boundary(text: str) -> re.Match[str] | None:
    """Nhận diện ``Điều`` chỉ khi pattern nằm ở đầu paragraph/block."""

    return ARTICLE_RE.match(text)


def detect_clause_boundary(text: str, inside_article: bool) -> re.Match[str] | None:
    """Nhận diện ``1.``/``2)`` chỉ trong context một Điều để tránh split nhầm."""

    return CLAUSE_RE.match(text) if inside_article else None


def detect_point_boundary(text: str, inside_article: bool) -> re.Match[str] | None:
    """Nhận diện ``a)``/``b.`` chỉ trong context Điều/Khoản."""

    return POINT_RE.match(text) if inside_article else None


def is_legal_basis(text: str) -> bool:
    """Kiểm tra paragraph căn cứ/theo đề nghị với regex tolerant lỗi OCR nhẹ."""

    return bool(LEGAL_BASIS_RE.match(text))


def is_recipient_boundary(text: str) -> bool:
    """Kiểm tra boundary ``Nơi nhận`` ở đầu paragraph."""

    return bool(RECIPIENT_RE.match(text))


def is_signature_boundary(text: str, block_type: str = "text") -> bool:
    """Nhận diện vùng chữ ký từ chức vụ rõ ràng hoặc figure caption có tên/chức vụ."""

    if SIGNATURE_RE.match(text):
        return True
    if block_type == "figure_caption":
        words = text.split()
        return 2 <= len(words) <= 12 and any(char.isalpha() for char in text)
    return False


def build_document_tree(
    blocks: Iterable[OCRBlock],
    document_id: str,
    document_title: str | None = None,
) -> DocumentTree:
    """Xây cây document/preamble/Điều/Khoản/Điểm/table/recipients/signature.

    Parser chạy tuần tự theo reading order. Mọi block sau một Điều được gắn vào
    Điều hiện tại cho tới Điều mới, ``Nơi nhận`` hoặc vùng chữ ký.
    """

    ordered = sort_blocks_in_reading_order(block for block in blocks if is_indexable_block(block))
    nodes: dict[str, DocumentNode] = {}
    ordinal = 0

    def add_node(
        node_type: str,
        title: str | None,
        raw_text: str,
        normalized_text: str,
        page: int,
        parent_id: str | None,
        source_block_id: str | None,
        metadata: dict | None = None,
    ) -> DocumentNode:
        """Tạo node deterministic và nối quan hệ parent/child hai chiều."""

        nonlocal ordinal
        ordinal += 1
        source = source_block_id or f"virtual-{ordinal}"
        node_id = _stable_node_id(document_id, node_type, source, ordinal)
        parent = nodes.get(parent_id) if parent_id else None
        section_path = list(parent.section_path) if parent else []
        if title and node_type not in {
            "document", "paragraph", "legal_basis", "recipient_item"
        }:
            section_path.append(title)
        node = DocumentNode(
            node_id=node_id,
            node_type=node_type,
            title=title,
            text_raw=raw_text.strip(),
            text_normalized=normalized_text.strip(),
            page_start=page,
            page_end=page,
            parent_id=parent_id,
            children_ids=[],
            section_path=section_path,
            source_block_ids=[source_block_id] if source_block_id else [],
            metadata=dict(metadata or {}),
        )
        nodes[node_id] = node
        if parent:
            parent.children_ids.append(node_id)
        return node

    first_page = ordered[0].page_number if ordered else 1
    root = add_node(
        "document",
        document_title or "Tài liệu",
        "",
        "",
        first_page,
        None,
        None,
        {"document_id": document_id},
    )

    metadata_node: DocumentNode | None = None
    preamble_node: DocumentNode | None = None
    decision_node: DocumentNode | None = None
    current_article: DocumentNode | None = None
    current_clause: DocumentNode | None = None
    current_point: DocumentNode | None = None
    recipients_node: DocumentNode | None = None
    signature_node: DocumentNode | None = None
    mode = "metadata"
    
    for block in ordered:
        segments = [(block.content_raw, block.content_normalized)]
        if block.block_type != "table":
            segments = _paragraph_segments(block.content_raw, block.content_normalized)

        for segment_index, (raw_text, text) in enumerate(segments):
            if not text:
                continue
            source_id = block.block_id
            source_metadata = {
                "block_type": block.block_type,
                "segment_index": segment_index,
                "bbox": block.bbox,
            }

            article_match = detect_article_boundary(text)
            if article_match:
                label = f"Điều {article_match.group(1)}"
                parent_id = decision_node.node_id if decision_node else root.node_id
                current_article = add_node(
                    "article", label, raw_text, text, block.page_number, parent_id,
                    source_id, {**source_metadata, "article_number": article_match.group(1)},
                )
                current_clause = None
                current_point = None
                recipients_node = None
                signature_node = None
                mode = "article"
                continue

            if is_recipient_boundary(text):
                recipients_node = add_node(
                    "recipients", "Nơi nhận", raw_text, text, block.page_number,
                    root.node_id, source_id, source_metadata,
                )
                current_article = None
                current_clause = None
                current_point = None
                signature_node = None
                mode = "recipients"
                continue

            if is_signature_boundary(text, block.block_type):
                if signature_node is None:
                    signature_node = add_node(
                        "signature", "Chữ ký", raw_text, text, block.page_number,
                        root.node_id, source_id, source_metadata,
                    )
                else:
                    _append_text(signature_node, raw_text, text, block.page_number, source_id)
                current_article = None
                current_clause = None
                current_point = None
                mode = "signature"
                continue

            if DECISION_RE.match(text):
                decision_node = add_node(
                    "decision_heading", "QUYẾT ĐỊNH", raw_text, text, block.page_number,
                    root.node_id, source_id, source_metadata,
                )
                current_article = None
                current_clause = None
                current_point = None
                mode = "decision"
                continue

            if is_legal_basis(text) and current_article is None:
                if preamble_node is None:
                    preamble_node = add_node(
                        "preamble", "Phần căn cứ", "", "", block.page_number,
                        root.node_id, None,
                    )
                add_node(
                    "legal_basis", None, raw_text, text, block.page_number,
                    preamble_node.node_id, source_id, source_metadata,
                )
                mode = "preamble"
                continue

            if block.block_type == "table":
                structural_parent = _active_structural_parent(
                    current_article, decision_node, preamble_node, metadata_node, root
                )
                add_node(
                    "table", "Bảng", raw_text, text, block.page_number,
                    structural_parent.node_id, source_id, source_metadata,
                )
                continue

            if mode == "recipients" and recipients_node:
                add_node(
                    "recipient_item", None, raw_text, text, block.page_number,
                    recipients_node.node_id, source_id, source_metadata,
                )
                continue

            if mode == "signature" and signature_node:
                _append_text(signature_node, raw_text, text, block.page_number, source_id)
                continue

            if current_article:
                clause_match = detect_clause_boundary(text, inside_article=True)
                point_match = detect_point_boundary(text, inside_article=True)
                if clause_match:
                    current_clause = add_node(
                        "clause", f"Khoản {clause_match.group(1)}", raw_text, text,
                        block.page_number, current_article.node_id, source_id,
                        {**source_metadata, "clause_number": clause_match.group(1)},
                    )
                    current_point = None
                elif point_match:
                    point_parent = current_clause or current_article
                    current_point = add_node(
                        "point", f"Điểm {point_match.group(1)}", raw_text, text,
                        block.page_number, point_parent.node_id, source_id,
                        {**source_metadata, "point_label": point_match.group(1)},
                    )
                else:
                    paragraph_parent = current_point or current_clause or current_article
                    add_node(
                        "paragraph", None, raw_text, text, block.page_number,
                        paragraph_parent.node_id, source_id, source_metadata,
                    )
                continue

            if mode in {"preamble", "decision"}:
                parent = preamble_node if mode == "preamble" and preamble_node else decision_node
                if parent:
                    add_node(
                        "paragraph", None, raw_text, text, block.page_number,
                        parent.node_id, source_id, source_metadata,
                    )
                    continue

            if metadata_node is None:
                metadata_node = add_node(
                    "metadata", "Thông tin văn bản", "", "", block.page_number,
                    root.node_id, None,
                )
            add_node(
                "paragraph", None, raw_text, text, block.page_number,
                metadata_node.node_id, source_id, source_metadata,
            )

    _propagate_page_ranges(tree=DocumentTree(root.node_id, nodes))
    return DocumentTree(root.node_id, nodes)


def iter_nodes(tree: DocumentTree, node_type: str | None = None) -> Iterator[DocumentNode]:
    """Duyệt depth-first theo thứ tự tài liệu, có thể lọc theo node type."""

    stack = [tree.root_id]
    while stack:
        node_id = stack.pop()
        node = tree.nodes[node_id]
        if node_type is None or node.node_type == node_type:
            yield node
        stack.extend(reversed(node.children_ids))


def find_ancestor(
    tree: DocumentTree,
    node_id: str,
    accepted_types: set[str],
) -> DocumentNode | None:
    """Đi ngược parent chain để tìm ancestor gần nhất thuộc nhóm type yêu cầu."""

    current = tree.nodes.get(node_id)
    while current:
        if current.node_type in accepted_types:
            return current
        current = tree.nodes.get(current.parent_id) if current.parent_id else None
    return None


def map_source_blocks_to_nodes(tree: DocumentTree) -> dict[str, DocumentNode]:
    """Tạo bảng tra source block -> node sâu nhất để gắn table vào đúng Điều."""

    result: dict[str, DocumentNode] = {}
    for node in iter_nodes(tree):
        for source_id in node.source_block_ids:
            result[source_id] = node
    return result


def _paragraph_segments(raw_text: str, normalized_text: str) -> list[tuple[str, str]]:
    """Tách OCR block theo newline để boundary ở đầu paragraph có thể được nhận diện."""

    raw_lines = [line.strip() for line in raw_text.replace("\r", "\n").split("\n") if line.strip()]
    normalized_lines = [line.strip() for line in normalized_text.split("\n") if line.strip()]
    if len(raw_lines) == len(normalized_lines) and normalized_lines:
        return list(zip(raw_lines, normalized_lines, strict=True))
    return [(raw_text.strip(), normalized_text.strip())]


def _active_structural_parent(
    article: DocumentNode | None,
    decision: DocumentNode | None,
    preamble: DocumentNode | None,
    metadata: DocumentNode | None,
    root: DocumentNode,
) -> DocumentNode:
    """Chọn structural parent gần nhất cho table/paragraph đặc biệt."""

    return article or decision or preamble or metadata or root


def _append_text(
    node: DocumentNode,
    raw_text: str,
    normalized_text: str,
    page_number: int,
    source_block_id: str,
) -> None:
    """Nối text vào node vùng (chữ ký) và cập nhật provenance/page range."""

    node.text_raw = "\n".join(part for part in (node.text_raw, raw_text.strip()) if part)
    node.text_normalized = "\n".join(
        part for part in (node.text_normalized, normalized_text.strip()) if part
    )
    node.page_end = max(node.page_end, page_number)
    if source_block_id not in node.source_block_ids:
        node.source_block_ids.append(source_block_id)


def _stable_node_id(document_id: str, node_type: str, source: str, ordinal: int) -> str:
    """Sinh UUID5 node ID ổn định kể cả document_id không phải UUID hợp lệ."""

    try:
        namespace = uuid.UUID(str(document_id))
    except (ValueError, TypeError, AttributeError):
        namespace = uuid.uuid5(uuid.NAMESPACE_URL, str(document_id))
    return str(uuid.uuid5(namespace, f"v2:node:{node_type}:{source}:{ordinal}"))


def _propagate_page_ranges(tree: DocumentTree) -> None:
    """Lan page range/source IDs từ descendants lên parent để lưu parent context."""

    for node in reversed(list(iter_nodes(tree))):
        for child_id in node.children_ids:
            child = tree.nodes[child_id]
            node.page_start = min(node.page_start, child.page_start)
            node.page_end = max(node.page_end, child.page_end)
            for source_id in child.source_block_ids:
                if source_id not in node.source_block_ids:
                    node.source_block_ids.append(source_id)
