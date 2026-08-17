"""Xây contextual retrieval text và Qdrant payload cho child/parent chunk."""

from __future__ import annotations

from typing import Any

from .models import Chunk


def safe_meta(value: Any) -> str:
    """Chuyển metadata nullable thành chuỗi sạch, không bao giờ tạo chữ ``None``."""

    if value is None:
        return ""
    return str(value).strip()


def build_context_lines(document_meta: dict[str, Any], section_path: list[str]) -> list[str]:
    """Tạo prefix metadata ngắn, chỉ thêm field thực sự có giá trị."""

    field_specs = (
        ("Văn bản", document_meta.get("Summary") or document_meta.get("summary")),
        ("Số", document_meta.get("No") or document_meta.get("no")),
        ("Cơ quan ban hành", document_meta.get("Author") or document_meta.get("author")),
        ("Ngày", document_meta.get("DateDocument") or document_meta.get("date")),
    )
    lines = [f"{label}: {safe_meta(value)}" for label, value in field_specs if safe_meta(value)]
    path = " > ".join(part for part in (safe_meta(item) for item in section_path) if part)
    if path:
        lines.append(f"Phần: {path}")
    return lines


def build_retrieval_text(document_meta: dict[str, Any], chunk: Chunk) -> str:
    """Ghép metadata/section path với normalized body để embedding có đủ context."""

    context = build_context_lines(document_meta, chunk.section_path)
    body = chunk.normalized_text.strip()
    if not body:
        return "\n".join(context).strip()
    return "\n".join([*context, "", "Nội dung:", body]).strip()


def chunk_to_payload(chunk: Chunk, document_meta: dict[str, Any]) -> dict[str, Any]:
    """Chuyển Chunk thành payload v2 đầy đủ provenance và field backward-compatible."""

    return {
        **document_meta,
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "chunk_type": chunk.chunk_type,
        "parent_id": chunk.parent_id,
        "section_path": list(chunk.section_path),
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "raw_text": chunk.raw_text,
        "normalized_text": chunk.normalized_text,
        "retrieval_text": chunk.retrieval_text,
        "token_count": chunk.token_count,
        "source_block_ids": list(chunk.source_block_ids),
        "source": "OCR_JSON",
        "chunking_version": "v2",
        **chunk.metadata,
    }

