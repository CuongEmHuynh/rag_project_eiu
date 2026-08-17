"""Parse HTML table, flatten header, merge bảng qua trang và serialize từng row."""

from __future__ import annotations

import copy
import logging
import re
import uuid
from collections.abc import Iterable
from typing import Any, Protocol

from bs4 import BeautifulSoup, Tag

from .models import DocumentTree, OCRBlock, ParsedTable
from .normalize import normalize_ocr_text, sort_blocks_in_reading_order
from .retrieval_text import build_context_lines
from .structure_parser import find_ancestor, map_source_blocks_to_nodes


LOGGER = logging.getLogger(__name__)
_COURSE_CODE_RE = re.compile(r"^[A-ZĐ]{2,}[ -]?\d{2,4}[A-Z]?$", re.IGNORECASE)
_GENERIC_HEADER_RE = re.compile(r"^Cột\s+\d+$", re.IGNORECASE)


class TableParseError(ValueError):
    """Lỗi HTML table không đủ cell/row để tạo semantic representation."""


class TableRowSerializer(Protocol):
    """Contract strategy serializer cho một row bất kỳ."""

    def serialize(self, table: ParsedTable, row: list[str]) -> str:
        """Serialize row thành semantic body, không thêm document prefix."""


class GenericKeyValueTableSerializer:
    """Fallback serializer ``header: value`` cho mọi table schema."""

    def serialize(self, table: ParsedTable, row: list[str]) -> str:
        """Ghép từng giá trị có thật với header tương ứng, không invent ô trống."""

        lines: list[str] = []
        for index in range(max(table.column_count, len(row))):
            value = normalize_ocr_text(row[index]) if index < len(row) else ""
            if not value:
                continue
            header = table.headers[index] if index < len(table.headers) else f"Cột {index + 1}"
            lines.append(f"{header}: {value}")
        return "\n".join(lines)


class CourseTransferTableSerializer:
    """Domain serializer phân biệt môn nguồn và môn được chuyển."""

    SOURCE_LABELS = ("Mã môn", "Tên môn", "Số tín chỉ", "Điểm")
    TARGET_LABELS = ("Mã môn chuyển", "Tên môn chuyển", "Số tín chỉ được chuyển", "Điểm chuyển đổi")

    def serialize(self, table: ParsedTable, row: list[str]) -> str:
        """Serialize tối đa 8 field course mapping bằng label nguồn/đích rõ nghĩa."""

        column_map = table.metadata.get("course_column_map") or list(range(min(8, len(row))))
        values = [row[index] if index < len(row) else "" for index in column_map[:8]]
        values.extend([""] * (8 - len(values)))

        sections: list[str] = []
        source_lines = _label_value_lines(self.SOURCE_LABELS, values[:4])
        target_lines = _label_value_lines(self.TARGET_LABELS, values[4:8])
        if source_lines:
            sections.append("Môn học đã học:\n" + "\n".join(source_lines))
        if target_lines:
            sections.append("Môn học được chuyển:\n" + "\n".join(target_lines))
        return "\n\n".join(sections)


def parse_html_table(
    html: str,
    table_id: str | None = None,
    page_number: int = 1,
    source_block_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ParsedTable:
    """Parse HTML table, mở rộng rowspan/colspan và flatten multi-row header.

    Nếu ``thead`` có vẻ là data row (trường hợp bảng tiếp tục qua trang), row đó
    được đưa về data và header tạm ``Cột n`` để schema trang trước có thể kế thừa.
    """

    soup = BeautifulSoup(html or "", "html.parser")
    table_tag = soup.find("table")
    if not isinstance(table_tag, Tag):
        raise TableParseError("Không tìm thấy thẻ <table>")
    tr_tags = table_tag.find_all("tr")
    if not tr_tags:
        raise TableParseError("Table không có <tr>")

    expanded = _expand_html_rows(tr_tags)
    if not expanded or max((len(row) for row in expanded), default=0) == 0:
        raise TableParseError("Table không có cell có thể parse")
    width = max(len(row) for row in expanded)
    expanded = [_pad_row(row, width) for row in expanded]

    header_indexes = _header_row_indexes(tr_tags)
    header_rows = [expanded[index] for index in header_indexes]
    data_rows = [row for index, row in enumerate(expanded) if index not in header_indexes]
    suspected_data_header = bool(header_rows and all(_looks_like_data_row(row) for row in header_rows))
    if suspected_data_header:
        data_rows = [*header_rows, *data_rows]
        header_rows = []

    headers = _flatten_headers(header_rows, width)
    table_metadata = dict(metadata or {})
    table_metadata["suspected_data_header"] = suspected_data_header
    table_metadata["raw_header_rows"] = header_rows
    table_metadata["row_source_block_ids"] = [list(source_block_ids or []) for _ in data_rows]
    schema_type, course_column_map = _detect_course_transfer_schema(headers)
    table_metadata["schema_type"] = schema_type
    if course_column_map:
        table_metadata["course_column_map"] = course_column_map

    source_ids = list(source_block_ids or [])
    logical_id = table_id or _stable_table_id(source_ids, page_number, html)
    return ParsedTable(
        table_id=logical_id,
        page_start=page_number,
        page_end=page_number,
        headers=headers,
        rows=[row for row in data_rows if any(cell.strip() for cell in row)],
        column_count=width,
        source_block_ids=source_ids,
        metadata=table_metadata,
    )


def parse_document_tables(
    blocks: Iterable[OCRBlock],
    tree: DocumentTree,
) -> list[ParsedTable]:
    """Parse mọi table block và gắn structural parent/section path từ document tree."""

    node_by_source = map_source_blocks_to_nodes(tree)
    parsed: list[ParsedTable] = []
    for block in sort_blocks_in_reading_order(blocks):
        if block.block_type != "table" or not block.content_normalized:
            continue
        table_node = node_by_source.get(block.block_id)
        parent = None
        if table_node:
            parent = find_ancestor(
                tree,
                table_node.parent_id or table_node.node_id,
                {"article", "preamble", "decision_heading", "metadata", "document"},
            )
        section_path = list(parent.section_path if parent else tree.root.section_path)
        section_path.append("Bảng")
        try:
            table = parse_html_table(
                block.content_raw,
                table_id=_stable_table_id([block.block_id], block.page_number, block.content_raw),
                page_number=block.page_number,
                source_block_ids=[block.block_id],
                metadata={
                    "bbox": block.bbox,
                    "structural_parent_id": parent.node_id if parent else tree.root_id,
                    "table_node_id": table_node.node_id if table_node else None,
                    "section_path": section_path,
                },
            )
        except TableParseError as exc:
            LOGGER.warning("Không parse được table %s: %s", block.block_id, exc)
            continue
        parsed.append(table)
    return parsed


def score_table_continuation(
    previous_table: ParsedTable,
    next_table: ParsedTable,
    previous_page_blocks: list[OCRBlock],
    next_page_blocks: list[OCRBlock],
) -> tuple[float, list[str]]:
    """Chấm điểm bảng trang kế tiếp dựa trên position/schema/structural boundary."""

    reasons: list[str] = []
    if next_table.page_start != previous_table.page_end + 1:
        return (-10.0, ["pages_not_adjacent"])
    score = 0.0
    previous_bbox = previous_table.metadata.get("bbox")
    next_bbox = next_table.metadata.get("bbox")
    if previous_bbox and float(previous_bbox[3]) > 0.80:
        score += 1.25
        reasons.append("previous_near_page_bottom")
    if next_bbox and float(next_bbox[1]) < 0.20:
        score += 1.25
        reasons.append("next_near_page_top")

    column_delta = abs(previous_table.column_count - next_table.column_count)
    if column_delta == 0:
        score += 1.5
        reasons.append("same_column_count")
    elif column_delta == 1:
        score += 0.5
        reasons.append("near_column_count")
    else:
        score -= 3.0
        reasons.append("different_column_count")

    same_parent = (
        previous_table.metadata.get("structural_parent_id")
        == next_table.metadata.get("structural_parent_id")
    )
    if same_parent:
        score += 1.5
        reasons.append("same_structural_parent")
    else:
        score -= 2.0
        reasons.append("different_structural_parent")

    boundaries = _boundaries_before_table(next_table, next_page_blocks)
    if boundaries:
        score -= 5.0
        reasons.extend(f"new_{boundary}_before_table" for boundary in boundaries)
    else:
        score += 1.0
        reasons.append("no_new_boundary_before_table")

    similarity = _header_similarity(previous_table.headers, next_table.headers)
    if similarity >= 0.6:
        score += 1.0
        reasons.append("similar_schema")
    elif next_table.metadata.get("suspected_data_header") or all(
        _GENERIC_HEADER_RE.match(header) for header in next_table.headers
    ):
        score += 0.75
        reasons.append("next_header_looks_like_data")
    elif similarity < 0.2:
        score -= 1.5
        reasons.append("new_semantic_schema")
    return score, reasons


def is_table_continuation(
    previous_table: ParsedTable,
    next_table: ParsedTable,
    previous_page_blocks: list[OCRBlock],
    next_page_blocks: list[OCRBlock],
    threshold: float = 3.0,
) -> bool:
    """Trả ``True`` khi điểm continuation đạt threshold cấu hình."""

    score, _ = score_table_continuation(
        previous_table, next_table, previous_page_blocks, next_page_blocks
    )
    return score >= threshold


def merge_table_continuation(previous_table: ParsedTable, next_table: ParsedTable) -> ParsedTable:
    """Tạo logical table mới, kế thừa schema trang trước và nối rows/provenance."""

    merged = copy.deepcopy(previous_table)
    merged.page_end = next_table.page_end
    merged.rows.extend(next_table.rows)
    for source_id in next_table.source_block_ids:
        if source_id not in merged.source_block_ids:
            merged.source_block_ids.append(source_id)
    merged.metadata["cross_page"] = True
    merged.metadata.setdefault("continuation_table_ids", []).append(next_table.table_id)
    merged.metadata["continuation_count"] = len(merged.metadata["continuation_table_ids"])
    merged.metadata.setdefault("row_source_block_ids", []).extend(
        copy.deepcopy(next_table.metadata.get("row_source_block_ids", []))
    )
    if not merged.headers or all(_GENERIC_HEADER_RE.match(header) for header in merged.headers):
        merged.headers = list(next_table.headers)
    return merged


def reconstruct_cross_page_tables(
    tables: Iterable[ParsedTable],
    blocks_by_page: dict[int, list[OCRBlock]],
    threshold: float = 3.0,
) -> list[ParsedTable]:
    """Ghép tuần tự các table part ở hai trang kề nhau thành logical table."""

    ordered = sorted(tables, key=lambda table: (table.page_start, table.table_id))
    logical_tables: list[ParsedTable] = []
    for table in ordered:
        if not logical_tables:
            logical_tables.append(copy.deepcopy(table))
            continue
        previous = logical_tables[-1]
        if is_table_continuation(
            previous,
            table,
            blocks_by_page.get(previous.page_end, []),
            blocks_by_page.get(table.page_start, []),
            threshold=threshold,
        ):
            logical_tables[-1] = merge_table_continuation(previous, table)
        else:
            logical_tables.append(copy.deepcopy(table))
    return logical_tables


def choose_table_serializer(table: ParsedTable) -> TableRowSerializer:
    """Chọn course-transfer strategy khi schema đủ tin cậy, ngược lại dùng generic."""

    if table.metadata.get("schema_type") == "course_transfer":
        return CourseTransferTableSerializer()
    return GenericKeyValueTableSerializer()


def serialize_table_row(
    table: ParsedTable,
    row: list[str],
    document_meta: dict[str, Any],
    section_path: list[str],
    include_context: bool = True,
) -> str:
    """Serialize một row bằng strategy phù hợp và tùy chọn document context.

    ``include_context=False`` được chunk builder dùng để tránh prepend metadata hai
    lần; API mặc định ``True`` tiện cho debug/benchmark trực tiếp.
    """

    body = choose_table_serializer(table).serialize(table, row).strip()
    if not include_context:
        return body
    context = build_context_lines(document_meta, section_path)
    return "\n".join([*context, "", "Nội dung:", body]).strip() if body else "\n".join(context)


def _expand_html_rows(rows: list[Tag]) -> list[list[str]]:
    """Mở rộng cell rowspan/colspan thành ma trận chữ nhật theo toạ độ logical."""

    occupied: dict[tuple[int, int], str] = {}
    max_column = 0
    for row_index, tr_tag in enumerate(rows):
        column_index = 0
        for cell in tr_tag.find_all(["th", "td"], recursive=False):
            while (row_index, column_index) in occupied:
                column_index += 1
            text = normalize_ocr_text(cell.get_text(" ", strip=True))
            rowspan = _positive_span(cell.get("rowspan"))
            colspan = _positive_span(cell.get("colspan"))
            for row_offset in range(rowspan):
                for column_offset in range(colspan):
                    occupied[(row_index + row_offset, column_index + column_offset)] = text
            column_index += colspan
            max_column = max(max_column, column_index)
    return [
        [occupied.get((row_index, column_index), "") for column_index in range(max_column)]
        for row_index in range(len(rows))
    ]


def _header_row_indexes(rows: list[Tag]) -> set[int]:
    """Xác định header rows từ ``thead`` hoặc chuỗi row toàn ``th`` đầu bảng."""

    indexes = {
        index for index, row in enumerate(rows)
        if row.find_parent("thead") is not None
    }
    if indexes:
        return indexes
    for index, row in enumerate(rows):
        direct_cells = row.find_all(["th", "td"], recursive=False)
        if direct_cells and all(cell.name == "th" for cell in direct_cells):
            indexes.add(index)
        else:
            break
    return indexes


def _flatten_headers(header_rows: list[list[str]], width: int) -> list[str]:
    """Ghép các level header theo cột, bỏ label lặp do rowspan/colspan expansion."""

    headers: list[str] = []
    for column_index in range(width):
        path: list[str] = []
        for row in header_rows:
            value = row[column_index].strip() if column_index < len(row) else ""
            if value and (not path or path[-1].casefold() != value.casefold()):
                path.append(value)
        headers.append(" > ".join(path) if path else f"Cột {column_index + 1}")
    return headers


def _looks_like_data_row(row: list[str]) -> bool:
    """Phân biệt data row bị OCR nhét vào ``thead`` với semantic header thật."""

    non_empty = [cell.strip() for cell in row if cell.strip()]
    if not non_empty:
        return False
    course_codes = sum(bool(_COURSE_CODE_RE.match(cell)) for cell in non_empty)
    numeric_or_grade = sum(
        bool(re.fullmatch(r"(?:\d+(?:[.,]\d+)?|[A-F][+\-.]?)", cell, re.IGNORECASE))
        for cell in non_empty
    )
    return course_codes >= 1 and course_codes + numeric_or_grade >= max(2, len(non_empty) // 3)


def _detect_course_transfer_schema(headers: list[str]) -> tuple[str, list[int]]:
    """Nhận diện schema chuyển môn và trả thứ tự 8 cột source/target."""

    combined = " ".join(headers).casefold()
    if len(headers) < 8 or "môn" not in combined or not any(
        marker in combined for marker in ("chuyển", "đã học", "sv")
    ):
        return "generic", []
    if len(headers) == 8:
        return "course_transfer", list(range(8))

    candidates = [index for index, header in enumerate(headers) if "stt" not in header.casefold()]
    if len(candidates) >= 8:
        return "course_transfer", candidates[:8]
    return "generic", []


def _boundaries_before_table(table: ParsedTable, page_blocks: list[OCRBlock]) -> set[str]:
    """Tìm Article/title/recipients xuất hiện trước table ở trang continuation."""

    source_ids = set(table.source_block_ids)
    boundaries: set[str] = set()
    for block in sort_blocks_in_reading_order(page_blocks):
        if block.block_id in source_ids:
            break
        text = block.content_normalized
        if re.match(r"^\s*Điều\s+\d+", text, re.IGNORECASE):
            boundaries.add("article")
        elif re.match(r"^\s*Nơi\s+nhận", text, re.IGNORECASE):
            boundaries.add("recipients")
        elif block.block_type == "title" and re.match(
            r"^\s*(?:QUY[ẾE]T\s+ĐỊNH|CHƯƠNG\b|PHẦN\b|MỤC\b|DANH\s+SÁCH\b)",
            text,
            re.IGNORECASE,
        ):
            boundaries.add("title")
    return boundaries


def _header_similarity(left: list[str], right: list[str]) -> float:
    """Tính Jaccard token similarity giữa hai flattened schemas."""

    left_tokens = set(re.findall(r"\w+", " ".join(left).casefold()))
    right_tokens = set(re.findall(r"\w+", " ".join(right).casefold()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _label_value_lines(labels: tuple[str, ...], values: list[str]) -> list[str]:
    """Tạo bullet chỉ cho giá trị có thật để không hallucinate missing cell."""

    return [
        f"- {label}: {normalize_ocr_text(value)}"
        for label, value in zip(labels, values, strict=True)
        if normalize_ocr_text(value)
    ]


def _positive_span(value: Any) -> int:
    """Parse rowspan/colspan và fallback về 1 khi HTML lỗi."""

    try:
        return max(1, int(value or 1))
    except (TypeError, ValueError):
        return 1


def _pad_row(row: list[str], width: int) -> list[str]:
    """Pad row ngắn bằng chuỗi rỗng, không phát minh dữ liệu."""

    return [*row, *([""] * max(0, width - len(row)))]


def _stable_table_id(source_ids: list[str], page_number: int, html: str) -> str:
    """Sinh logical table UUID5 từ provenance thay vì index toàn document."""

    signature = ",".join(source_ids) or normalize_ocr_text(html)[:200]
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"sahc-v2:table:{page_number}:{signature}"))
