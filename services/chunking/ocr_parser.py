"""Đọc OCR JSON và chuyển nhiều biến thể schema thành ``OCRBlock`` thống nhất."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import OCRBlock, OCRDocument
from .normalize import normalize_block_type, normalize_ocr_text


class OCRParseError(ValueError):
    """Lỗi OCR JSON không đủ cấu trúc để chunk an toàn."""


def load_ocr_json(json_path: str | Path) -> OCRDocument:
    """Đọc UTF-8 JSON từ disk và gọi parser thuần ``parse_ocr_data``."""

    path = Path(json_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OCRParseError(f"Không tìm thấy OCR JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OCRParseError(f"OCR JSON không hợp lệ tại {path}: {exc}") from exc
    return parse_ocr_data(data)


def parse_ocr_data(data: dict[str, Any]) -> OCRDocument:
    """Parse dict OCR thành document và không làm mất content gốc.

    Hỗ trợ field chuẩn trong spec và một số alias thường gặp: ``page_id``,
    ``category_name``, ``label``, ``bbox_xyxy``, ``text`` và ``ocr``.
    """

    if not isinstance(data, dict):
        raise OCRParseError("OCR root phải là JSON object")
    pages = data.get("pages")
    if not isinstance(pages, list):
        raise OCRParseError("OCR JSON thiếu list 'pages'")

    parsed_blocks: list[OCRBlock] = []
    for page_offset, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            continue
        page_number = _parse_page_number(page, page_offset)
        width = _as_positive_float(page.get("width") or page.get("page_width"))
        height = _as_positive_float(page.get("height") or page.get("page_height"))
        blocks = page.get("blocks") or page.get("layout_blocks") or []
        if not isinstance(blocks, list):
            raise OCRParseError(f"pages[{page_offset - 1}].blocks phải là list")

        for offset, raw_block in enumerate(blocks, start=1):
            if not isinstance(raw_block, dict):
                continue
            block_index = _parse_block_index(raw_block, offset)
            raw_content = _extract_content(raw_block)
            bbox = _parse_bbox(raw_block, width, height)
            block_type = normalize_block_type(
                raw_block.get("type")
                or raw_block.get("category_name")
                or raw_block.get("label")
            )
            known_fields = {
                "type", "category_name", "label", "content", "content_raw", "text",
                "ocr", "bbox", "bbox_xyxy", "box", "angle", "rotation",
            }
            metadata = {key: value for key, value in raw_block.items() if key not in known_fields}
            metadata.update(
                {
                    "page_width": width,
                    "page_height": height,
                    "bbox_normalized": bool(bbox and width and height),
                }
            )
            parsed_blocks.append(
                OCRBlock(
                    page_number=page_number,
                    block_index=block_index,
                    block_type=block_type,
                    bbox=bbox,
                    content_raw=raw_content,
                    content_normalized=normalize_ocr_text(raw_content),
                    angle=raw_block.get("angle", raw_block.get("rotation")),
                    metadata=metadata,
                )
            )

    declared_count = data.get("page_count", data.get("num_pages", len(pages)))
    try:
        page_count = int(declared_count)
    except (TypeError, ValueError):
        page_count = len(pages)
    return OCRDocument(
        input_file=data.get("input_file") or data.get("input") or data.get("filename"),
        page_count=max(page_count, len(pages)),
        blocks=parsed_blocks,
        raw_data=data,
    )


def group_blocks_by_page(blocks: list[OCRBlock]) -> dict[int, list[OCRBlock]]:
    """Nhóm block theo page number để table continuation dùng context lân cận."""

    grouped: dict[int, list[OCRBlock]] = {}
    for block in blocks:
        grouped.setdefault(block.page_number, []).append(block)
    return grouped


def _parse_page_number(page: dict[str, Any], fallback: int) -> int:
    """Đọc page number tolerant với ID như ``page_001``."""

    value = page.get("page_number", page.get("page_id", fallback))
    try:
        return int(value)
    except (TypeError, ValueError):
        digits = "".join(char for char in str(value) if char.isdigit())
        return int(digits) if digits else fallback


def _parse_block_index(block: dict[str, Any], fallback: int) -> int:
    """Lấy index/order của block hoặc dùng thứ tự trong array."""

    value = block.get("block_index", block.get("order", block.get("reading_order", block.get("id"))))
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _extract_content(block: dict[str, Any]) -> str:
    """Lấy text gốc mà không ép object OCR phức tạp thành representation giả."""

    for key in ("content_raw", "content", "text"):
        value = block.get(key)
        if value is not None:
            return str(value)
    ocr_value = block.get("ocr")
    if isinstance(ocr_value, str):
        return ocr_value
    if isinstance(ocr_value, dict):
        for key in ("text", "content", "rec_text"):
            if ocr_value.get(key) is not None:
                return str(ocr_value[key])
    return ""


def _parse_bbox(
    block: dict[str, Any],
    width: float | None,
    height: float | None,
) -> tuple[float, float, float, float] | None:
    """Parse xyxy bbox và chuẩn hoá về 0..1 khi page dimensions khả dụng."""

    value = block.get("bbox") or block.get("bbox_xyxy") or block.get("box")
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    if width and height and max(abs(x1), abs(x2)) > 1.0:
        x1, x2 = x1 / width, x2 / width
        y1, y2 = y1 / height, y2 / height
    return (x1, y1, x2, y2)


def _as_positive_float(value: Any) -> float | None:
    """Chuyển page dimension sang float dương hoặc trả ``None``."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None
