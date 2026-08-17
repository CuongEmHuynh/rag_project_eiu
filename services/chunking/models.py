"""Các data model dùng xuyên suốt pipeline SAHC-v2.

Module này chỉ chứa dữ liệu, không chứa heuristic parsing. Việc tách riêng giúp các
stage của pipeline có contract rõ ràng và dễ unit test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class OCRBlock:
    """Một layout block được chuẩn hoá từ OCR JSON nhưng vẫn giữ nguyên nội dung gốc."""

    page_number: int
    block_index: int
    block_type: str
    bbox: tuple[float, float, float, float] | None
    content_raw: str
    content_normalized: str
    angle: float | int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def block_id(self) -> str:
        """Trả về ID ổn định theo trang/vị trí để audit nguồn của chunk."""

        return f"page_{self.page_number:03d}_block_{self.block_index:04d}"


@dataclass(slots=True)
class OCRDocument:
    """Kết quả parse OCR JSON cùng danh sách block đã chuẩn hoá."""

    input_file: str | None
    page_count: int
    blocks: list[OCRBlock]
    raw_data: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(slots=True)
class DocumentNode:
    """Một node trong cây cấu trúc tài liệu hành chính."""

    node_id: str
    node_type: str
    title: str | None
    text_raw: str
    text_normalized: str
    page_start: int
    page_end: int
    parent_id: str | None
    children_ids: list[str]
    section_path: list[str]
    source_block_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentTree:
    """Cây tài liệu dưới dạng root ID và bảng tra node theo ID."""

    root_id: str
    nodes: dict[str, DocumentNode]

    @property
    def root(self) -> DocumentNode:
        """Lấy root document node."""

        return self.nodes[self.root_id]


@dataclass(slots=True)
class ParsedTable:
    """Bảng logic sau khi HTML được mở rộng rowspan/colspan và flatten header."""

    table_id: str
    page_start: int
    page_end: int
    headers: list[str]
    rows: list[list[str]]
    column_count: int
    source_block_ids: list[str]
    continuation_of: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AtomicUnit:
    """Đơn vị semantic nhỏ nhất được bảo toàn trước bước token packing."""

    unit_id: str
    unit_type: str
    parent_id: str
    section_path: list[str]
    raw_text: str
    normalized_text: str
    page_start: int
    page_end: int
    source_block_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PackedUnit:
    """Một hoặc nhiều AtomicUnit đã được đóng gói trong cùng token budget."""

    units: list[AtomicUnit]
    raw_text: str
    normalized_text: str
    page_start: int
    page_end: int
    source_block_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """Record parent/child cuối cùng dùng để lưu trữ hoặc embedding."""

    chunk_id: str
    document_id: str
    parent_id: str | None
    chunk_index: int
    chunk_type: str
    section_path: list[str]
    page_start: int
    page_end: int
    raw_text: str
    normalized_text: str
    retrieval_text: str
    token_count: int
    source_block_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChunkingConfig:
    """Tập trung toàn bộ policy/magic number có thể điều chỉnh của SAHC-v2."""

    safety_margin_tokens: int = 16
    fallback_overlap_tokens: int = 20
    special_token_margin: int = 0
    index_recipients: bool = True
    index_signature: bool = True
    merge_cross_page_tables: bool = True
    prefer_single_table_row_chunks: bool = True
    enable_v1_txt_fallback: bool = True
    table_continuation_threshold: float = 3.0
    min_retrieval_tokens: int = 3


@dataclass(slots=True)
class RetrievalResult:
    """Kết quả search child kèm context parent/sibling sau expansion."""

    score: float | None
    child: dict[str, Any]
    context: list[dict[str, Any]] = field(default_factory=list)

