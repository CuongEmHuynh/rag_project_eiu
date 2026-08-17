"""Helpers tích hợp SAHC-v2 vào embedding/Qdrant mà không đụng production collection."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from .chunk_builder import build_document_chunks_v2
from .legacy import chunk_legal_document_v1, clean_data_v1
from .models import AtomicUnit, Chunk, ChunkingConfig
from .retrieval_text import build_retrieval_text, chunk_to_payload
from .token_counter import TokenCounter
from .token_packer import pack_atomic_units
from .validators import validate_chunks


LOGGER = logging.getLogger(__name__)
OCR_JSON_DIR = Path(os.getenv("OCR_JSON_DIR", "./data/file_contents"))
COLLECTION_NAME_V2 = os.getenv("COLLECTION_NAME_V2", "rag_document_v2")
CHUNKING_VERSION = os.getenv("CHUNKING_VERSION", "v2").lower()


def resolve_ocr_json_path(
    document_meta: dict[str, Any],
    ocr_json_dir: str | Path = OCR_JSON_DIR,
) -> Path:
    """Resolve một nơi duy nhất cho ``{Id}.json``, tolerant chữ hoa/thường của UUID."""

    document_id = str(document_meta.get("Id") or document_meta.get("id") or "").strip()
    if not document_id:
        raise ValueError("Metadata thiếu Id")
    directory = Path(ocr_json_dir)
    candidates = [
        directory / f"{document_id}.json",
        directory / f"{document_id.upper()}.json",
        directory / f"{document_id.lower()}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_v1_txt_path(
    document_meta: dict[str, Any],
    ocr_dir: str | Path = OCR_JSON_DIR,
) -> Path:
    """Resolve TXT fallback ở cùng OCR directory; không tự kích hoạt fallback."""

    document_id = str(document_meta.get("Id") or document_meta.get("id") or "").strip()
    if not document_id:
        raise ValueError("Metadata thiếu Id")
    directory = Path(ocr_dir)
    candidates = [
        directory / f"{document_id}.txt",
        directory / f"{document_id.upper()}.txt",
        directory / f"{document_id.lower()}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_v2_chunks(
    document_meta: dict[str, Any],
    embedding_model: Any,
    *,
    ocr_json_dir: str | Path = OCR_JSON_DIR,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Resolve OCR JSON rồi chạy v2; missing JSON được báo rõ, không silent dùng TXT."""

    path = resolve_ocr_json_path(document_meta, ocr_json_dir)
    if not path.exists():
        raise FileNotFoundError(f"Thiếu OCR JSON production: {path}")
    return build_document_chunks_v2(str(path), document_meta, embedding_model, config)


def load_chunks_by_version(
    document_meta: dict[str, Any],
    embedding_model: Any,
    *,
    version: str = CHUNKING_VERSION,
    ocr_dir: str | Path = OCR_JSON_DIR,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Route bằng feature flag ``v1|v2`` và chỉ fallback TXT khi config cho phép.

    Nếu v2 JSON thiếu nhưng TXT tồn tại, warning nêu rõ version ``v1-fallback`` để
    payload/index không bị trộn mà không quan sát được.
    """

    selected = version.strip().lower()
    policy = config or ChunkingConfig()
    if selected == "v1":
        txt_path = resolve_v1_txt_path(document_meta, ocr_dir)
        if not txt_path.exists():
            raise FileNotFoundError(f"Thiếu OCR TXT baseline: {txt_path}")
        return build_v1_fallback_chunks(txt_path, document_meta, embedding_model, policy)
    if selected != "v2":
        raise ValueError(f"CHUNKING_VERSION chỉ hỗ trợ v1 hoặc v2, nhận: {version}")

    json_path = resolve_ocr_json_path(document_meta, ocr_dir)
    if json_path.exists():
        return build_document_chunks_v2(str(json_path), document_meta, embedding_model, policy)
    txt_path = resolve_v1_txt_path(document_meta, ocr_dir)
    if policy.enable_v1_txt_fallback and txt_path.exists():
        LOGGER.warning(
            "Thiếu OCR JSON %s; dùng TXT fallback explicit với chunking_version=v1-fallback",
            json_path,
        )
        return build_v1_fallback_chunks(txt_path, document_meta, embedding_model, policy)
    raise FileNotFoundError(f"Thiếu OCR JSON production: {json_path}")


def build_v1_fallback_chunks(
    txt_path: str | Path,
    document_meta: dict[str, Any],
    embedding_model: Any,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Bọc baseline Điều-regex thành parent/child records và vẫn chặn token overflow.

    Đây là backward-compatibility path, không phải representation production v2.
    ``chunking_version=v1-fallback`` và ``source=OCR_TXT_FALLBACK`` luôn được ghi rõ.
    """

    policy = config or ChunkingConfig()
    counter = TokenCounter(embedding_model)
    path = Path(txt_path)
    clean_text = clean_data_v1(path.read_text(encoding="utf-8"))
    baseline = chunk_legal_document_v1(clean_text)
    document_id = str(document_meta.get("Id") or document_meta.get("id") or path.stem)
    parent_id = _stable_uuid(document_id, "v1-fallback:parent:document")

    units = [
        AtomicUnit(
            unit_id=f"v1-unit:{index}",
            unit_type=f"v1_{chunk_type}",
            parent_id="v1-fallback-root",
            section_path=["V1 fallback", f"Chunk {index}"],
            raw_text=text,
            normalized_text=text,
            page_start=1,
            page_end=1,
            source_block_ids=[f"ocr_txt_chunk_{index:04d}"],
            metadata={"baseline_chunk_index": index},
        )
        for index, (chunk_type, text) in enumerate(baseline)
    ]

    def candidate_builder(candidate: list[AtomicUnit]) -> str:
        """Đếm full contextual text cho baseline fallback, không cho model truncate."""

        first = candidate[0]
        draft = Chunk(
            chunk_id="draft",
            document_id=document_id,
            parent_id=parent_id,
            chunk_index=-1,
            chunk_type=first.unit_type,
            section_path=list(first.section_path),
            page_start=1,
            page_end=1,
            raw_text="\n\n".join(unit.raw_text for unit in candidate),
            normalized_text="\n\n".join(unit.normalized_text for unit in candidate),
            retrieval_text="",
            token_count=0,
            source_block_ids=[source for unit in candidate for source in unit.source_block_ids],
            metadata={},
        )
        return build_retrieval_text(document_meta, draft)

    packed = pack_atomic_units(units, counter, policy, candidate_builder)
    parent = Chunk(
        chunk_id=parent_id,
        document_id=document_id,
        parent_id=None,
        chunk_index=0,
        chunk_type="document",
        section_path=["V1 fallback"],
        page_start=1,
        page_end=1,
        raw_text=clean_text,
        normalized_text=clean_text,
        retrieval_text="",
        token_count=0,
        source_block_ids=["ocr_txt"],
        metadata={
            "record_type": "parent",
            "is_embedding_child": False,
            "chunking_version": "v1-fallback",
            "source": "OCR_TXT_FALLBACK",
            "child_ids": [],
        },
    )
    parent.retrieval_text = build_retrieval_text(document_meta, parent)
    parent.token_count = counter.count(parent.retrieval_text)

    children: list[Chunk] = []
    for index, item in enumerate(packed):
        first = item.units[0]
        child = Chunk(
            chunk_id=_stable_uuid(
                document_id,
                f"v1-fallback:child:{','.join(item.metadata['atomic_unit_ids'])}",
            ),
            document_id=document_id,
            parent_id=parent_id,
            chunk_index=index,
            chunk_type=first.unit_type,
            section_path=list(first.section_path),
            page_start=1,
            page_end=1,
            raw_text=item.raw_text,
            normalized_text=item.normalized_text,
            retrieval_text="",
            token_count=0,
            source_block_ids=list(item.source_block_ids),
            metadata={
                **item.metadata,
                "record_type": "child",
                "is_embedding_child": True,
                "chunking_version": "v1-fallback",
                "source": "OCR_TXT_FALLBACK",
            },
        )
        child.retrieval_text = build_retrieval_text(document_meta, child)
        child.token_count = counter.count(child.retrieval_text)
        children.append(child)
        parent.metadata["child_ids"].append(child.chunk_id)
    chunks = [parent, *children]
    validate_chunks(chunks, counter, policy)
    return chunks


def embedding_children(chunks: list[Chunk]) -> list[Chunk]:
    """Lọc đúng child records; parent không được encode/search bằng zero vector."""

    return [
        chunk for chunk in chunks
        if chunk.metadata.get("record_type") == "child"
        and chunk.metadata.get("is_embedding_child", True)
    ]


def build_parent_store(
    chunks: list[Chunk],
    document_meta: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Tạo JSON-serializable parent store riêng, tránh zero-vector parent trong Qdrant."""

    return {
        chunk.chunk_id: chunk_to_payload(chunk, document_meta)
        for chunk in chunks
        if chunk.metadata.get("record_type") == "parent"
    }


def build_qdrant_points(
    chunks: list[Chunk],
    document_meta: dict[str, Any],
    embedding_model: Any,
) -> list[Any]:
    """Encode child retrieval_text và tạo ``PointStruct`` với payload schema v2."""

    from qdrant_client.models import PointStruct

    counter = TokenCounter(embedding_model)
    validate_chunks(chunks, counter)
    points: list[Any] = []
    for chunk in embedding_children(chunks):
        vector = embedding_model.encode(chunk.retrieval_text, normalize_embeddings=True)
        vector_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        points.append(
            PointStruct(
                id=chunk.chunk_id,
                vector=vector_list,
                payload=chunk_to_payload(chunk, document_meta),
            )
        )
    return points


def upsert_document_v2(
    client: Any,
    chunks: list[Chunk],
    document_meta: dict[str, Any],
    embedding_model: Any,
    *,
    collection_name: str = COLLECTION_NAME_V2,
) -> int:
    """Upsert child points vào collection v2 đã tồn tại; không create/recreate/delete."""

    points = build_qdrant_points(chunks, document_meta, embedding_model)
    if points:
        client.upsert(collection_name=collection_name, points=points)
    return len(points)


def create_v2_collection_explicit(
    client: Any,
    embedding_model: Any,
    *,
    collection_name: str = COLLECTION_NAME_V2,
) -> None:
    """Tạo collection v2 khi operator gọi explicit; từ chối tên production v1 mặc định."""

    if collection_name == "rag_document":
        raise ValueError("Từ chối tạo/ghi collection production v1 'rag_document' bằng helper v2")
    from qdrant_client import models

    dimension_method = getattr(embedding_model, "get_sentence_embedding_dimension", None)
    if not callable(dimension_method):
        raise ValueError("Embedding model không công bố sentence embedding dimension")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=int(dimension_method()),
            distance=models.Distance.COSINE,
        ),
    )


def _stable_uuid(document_id: str, name: str) -> str:
    """Sinh UUID5 deterministic cho integration/fallback records."""

    try:
        namespace = uuid.UUID(document_id)
    except (ValueError, TypeError, AttributeError):
        namespace = uuid.uuid5(uuid.NAMESPACE_URL, document_id)
    return str(uuid.uuid5(namespace, name))
