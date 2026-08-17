"""Token-aware packing: giữ semantic boundary, token window chỉ là fallback cuối."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import replace

from .models import AtomicUnit, ChunkingConfig, PackedUnit
from .token_counter import TokenCounter


LOGGER = logging.getLogger(__name__)
CandidateTextBuilder = Callable[[list[AtomicUnit]], str]


class TokenPackingError(ValueError):
    """Không thể tạo chunk vừa model context mà không silent truncation."""


def pack_atomic_units(
    units: Iterable[AtomicUnit],
    token_counter: TokenCounter,
    config: ChunkingConfig | None = None,
    candidate_text_builder: CandidateTextBuilder | None = None,
) -> list[PackedUnit]:
    """Đóng gói AtomicUnit theo token budget và không trộn structural parent.

    Candidate được đếm trên *retrieval text hoàn chỉnh* qua callback, vì vậy prefix
    metadata/section path luôn nằm trong budget. Table row mặc định một row/chunk.
    """

    policy = config or ChunkingConfig()
    builder = candidate_text_builder or _default_candidate_text
    max_tokens = (
        token_counter.max_seq_length
        - policy.safety_margin_tokens
        - policy.special_token_margin
    )
    if max_tokens <= 0:
        raise TokenPackingError("Safety margin lớn hơn hoặc bằng max_seq_length")

    expanded: list[AtomicUnit] = []
    for unit in units:
        if token_counter.count(builder([unit])) <= max_tokens:
            expanded.append(unit)
        else:
            expanded.extend(
                split_oversized_atomic_unit(unit, token_counter, builder, policy, max_tokens)
            )

    packed: list[PackedUnit] = []
    current: list[AtomicUnit] = []
    for unit in expanded:
        force_single = policy.prefer_single_table_row_chunks and unit.unit_type == "table_row"
        if current and (force_single or not _can_pack_together(current[-1], unit)):
            packed.append(_to_packed_unit(current))
            current = []

        candidate = [*current, unit]
        if current and token_counter.count(builder(candidate)) > max_tokens:
            packed.append(_to_packed_unit(current))
            current = []
            candidate = [unit]

        if token_counter.count(builder(candidate)) > max_tokens:
            raise TokenPackingError(f"Unit {unit.unit_id} vẫn overflow sau recursive split")
        current = candidate
        if force_single:
            packed.append(_to_packed_unit(current))
            current = []

    if current:
        packed.append(_to_packed_unit(current))
    return packed


def split_oversized_atomic_unit(
    unit: AtomicUnit,
    token_counter: TokenCounter,
    candidate_text_builder: CandidateTextBuilder,
    config: ChunkingConfig,
    max_tokens: int | None = None,
) -> list[AtomicUnit]:
    """Split unit theo paragraph/sentence trước, token window là fallback cuối."""

    limit = max_tokens or (
        token_counter.max_seq_length - config.safety_margin_tokens - config.special_token_margin
    )
    semantic_parts, boundary = _semantic_parts(unit.normalized_text)
    if len(semantic_parts) > 1:
        output: list[AtomicUnit] = []
        for index, part in enumerate(semantic_parts):
            child = replace(
                unit,
                unit_id=f"{unit.unit_id}:semantic:{index}",
                raw_text=part,
                normalized_text=part,
                metadata={**unit.metadata, "split_fallback": boundary},
            )
            if token_counter.count(candidate_text_builder([child])) <= limit:
                output.append(child)
            else:
                output.extend(
                    _split_by_token_window(
                        child, token_counter, candidate_text_builder, config, limit
                    )
                )
        return output
    return _split_by_token_window(unit, token_counter, candidate_text_builder, config, limit)


def _split_by_token_window(
    unit: AtomicUnit,
    token_counter: TokenCounter,
    candidate_text_builder: CandidateTextBuilder,
    config: ChunkingConfig,
    max_tokens: int,
) -> list[AtomicUnit]:
    """Cắt token window có overlap; giảm window đến khi full retrieval text vừa budget."""

    token_ids = token_counter.encode_content(unit.normalized_text)
    if not token_ids:
        raise TokenPackingError(f"Unit {unit.unit_id} không có token để split")

    prefix_probe = replace(unit, raw_text="", normalized_text="")
    prefix_tokens = token_counter.count(candidate_text_builder([prefix_probe]))
    window_size = max_tokens - prefix_tokens
    if window_size <= 0:
        raise TokenPackingError(
            f"Context prefix của unit {unit.unit_id} đã vượt token budget ({prefix_tokens})"
        )

    overlap = min(config.fallback_overlap_tokens, max(0, window_size - 1))
    output: list[AtomicUnit] = []
    start = 0
    split_index = 0
    while start < len(token_ids):
        end = min(len(token_ids), start + window_size)
        candidate: AtomicUnit | None = None
        while end > start:
            part = token_counter.decode_content(token_ids[start:end]).strip()
            candidate = replace(
                unit,
                unit_id=f"{unit.unit_id}:tokens:{split_index}",
                raw_text=part,
                normalized_text=part,
                metadata={**unit.metadata, "split_fallback": "token_window"},
            )
            if token_counter.count(candidate_text_builder([candidate])) <= max_tokens:
                break
            end -= max(1, (end - start) // 8)
        if candidate is None or end <= start:
            raise TokenPackingError(f"Không thể fit dù chỉ một token của unit {unit.unit_id}")
        output.append(candidate)
        LOGGER.warning("Token-window fallback được dùng cho unit %s", unit.unit_id)
        split_index += 1
        if end >= len(token_ids):
            break
        next_start = end - overlap
        start = next_start if next_start > start else end
    return output


def _semantic_parts(text: str) -> tuple[list[str], str]:
    """Thử boundary theo paragraph rồi sentence; trả nguyên text nếu không tách được."""

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n|(?=^\s*\d+[.)]\s+)|(?=^\s*[a-zđ][.)]\s+)", text, flags=re.MULTILINE) if part.strip()]
    if len(paragraphs) > 1:
        return paragraphs, "paragraph"
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?;:])\s+(?=[A-ZÀ-Ỹ0-9])", text)
        if part.strip()
    ]
    if len(sentences) > 1:
        return sentences, "sentence"
    return [text.strip()], "none"


def _can_pack_together(left: AtomicUnit, right: AtomicUnit) -> bool:
    """Chỉ pack unit cùng parent/type/table để không phá semantic boundary."""

    return (
        left.parent_id == right.parent_id
        and left.unit_type == right.unit_type
        and left.section_path == right.section_path
        and left.metadata.get("table_id") == right.metadata.get("table_id")
    )


def _default_candidate_text(units: list[AtomicUnit]) -> str:
    """Builder mặc định dùng khi caller không có contextual retrieval prefix."""

    return "\n\n".join(unit.normalized_text for unit in units if unit.normalized_text)


def _to_packed_unit(units: list[AtomicUnit]) -> PackedUnit:
    """Gộp text/page/provenance và giữ metadata của unit đầu cho chunk builder."""

    source_ids: list[str] = []
    for unit in units:
        for source_id in unit.source_block_ids:
            if source_id not in source_ids:
                source_ids.append(source_id)
    metadata = dict(units[0].metadata)
    metadata["atomic_unit_ids"] = [unit.unit_id for unit in units]
    fallbacks = [unit.metadata.get("split_fallback") for unit in units if unit.metadata.get("split_fallback")]
    if fallbacks:
        metadata["split_fallback"] = fallbacks[-1]
    return PackedUnit(
        units=list(units),
        raw_text="\n\n".join(unit.raw_text for unit in units if unit.raw_text),
        normalized_text="\n\n".join(unit.normalized_text for unit in units if unit.normalized_text),
        page_start=min(unit.page_start for unit in units),
        page_end=max(unit.page_end for unit in units),
        source_block_ids=source_ids,
        metadata=metadata,
    )
