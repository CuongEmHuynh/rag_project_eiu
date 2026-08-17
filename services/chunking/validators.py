"""Validation gate bắt buộc trước khi chunk được embedding/upsert."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .models import Chunk, ChunkingConfig
from .token_counter import TokenCounter


class ChunkValidationError(ValueError):
    """Tập hợp một hoặc nhiều lỗi integrity/token của output SAHC-v2."""

    def __init__(self, errors: list[str]) -> None:
        """Lưu danh sách lỗi để test/log có thể inspect từng nguyên nhân."""

        self.errors = errors
        super().__init__("Chunk validation thất bại:\n- " + "\n- ".join(errors))


def validate_chunks(
    chunks: Iterable[Chunk],
    model_or_counter: Any,
    config: ChunkingConfig | None = None,
) -> None:
    """Kiểm tra stable ID, parent, table context, non-empty và token overflow.

    Token overflow chỉ áp dụng cho embedding children; parent store có thể dài hơn
    vì không được đưa vào vector search. Không có silent truncation.
    """

    policy = config or ChunkingConfig()
    counter = (
        model_or_counter
        if isinstance(model_or_counter, TokenCounter)
        else TokenCounter(model_or_counter)
    )
    items = list(chunks)
    errors: list[str] = []
    ids = [chunk.chunk_id for chunk in items]
    if len(ids) != len(set(ids)):
        errors.append("chunk_id bị trùng")

    parent_ids = {
        chunk.chunk_id
        for chunk in items
        if chunk.metadata.get("record_type") == "parent"
    }
    for chunk in items:
        is_child = chunk.metadata.get("record_type") == "child" or chunk.metadata.get(
            "is_embedding_child", False
        )
        if not is_child:
            continue
        if not chunk.parent_id or chunk.parent_id not in parent_ids:
            errors.append(f"{chunk.chunk_id}: missing parent_id hợp lệ")
        if not chunk.retrieval_text.strip():
            errors.append(f"{chunk.chunk_id}: retrieval_text rỗng")
            continue
        actual_tokens = counter.count(chunk.retrieval_text)
        if actual_tokens != chunk.token_count:
            errors.append(
                f"{chunk.chunk_id}: token_count={chunk.token_count}, thực tế={actual_tokens}"
            )
        if actual_tokens > counter.max_seq_length:
            errors.append(
                f"{chunk.chunk_id}: {actual_tokens} tokens vượt max_seq_length="
                f"{counter.max_seq_length}"
            )
        if actual_tokens < policy.min_retrieval_tokens:
            errors.append(f"{chunk.chunk_id}: retrieval_text quá ngắn ({actual_tokens} tokens)")
        if chunk.chunk_type == "table_row":
            if not chunk.metadata.get("table_id"):
                errors.append(f"{chunk.chunk_id}: table row thiếu table_id")
            if not chunk.section_path:
                errors.append(f"{chunk.chunk_id}: table row thiếu section_path")
    if errors:
        raise ChunkValidationError(errors)

