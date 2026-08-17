"""Chuẩn hoá text/block và lọc noise theo heuristic có thể audit."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from .models import OCRBlock


_PAGE_NUMBER_RE = re.compile(r"^\s*(?:trang\s*)?\d{1,4}\s*$", re.IGNORECASE)
_SHORT_OPERATOR_NOTE_RE = re.compile(
    r"^\s*[A-ZÀ-Ỹ][\wÀ-ỹ .'-]{0,28}\s*\((?:ĐT|DT|đt|dt)\)\s*$"
)
_VALUABLE_ABANDON_RE = re.compile(
    r"\b(?:BỘ|TRƯỜNG|ỦY BAN|CỘNG HÒA|Số\s*:|QUYẾT ĐỊNH|Căn cứ|Điều\s+\d+)\b",
    re.IGNORECASE,
)
_STAMP_NOISE_RE = re.compile(r"^(?:ký bởi|signed by|signature valid|digitally signed)\b", re.IGNORECASE)


def normalize_ocr_text(text: str) -> str:
    """Normalize nhẹ để parser ổn định mà không đoán/sửa lỗi OCR pháp lý.

    Hàm dùng Unicode NFC, đổi NBSP thành space, thu gọn whitespace ngang và tối
    đa hai newline. Các ký tự đáng ngờ như ``?`` hoặc lỗi chính tả OCR được giữ.
    """

    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", str(text)).replace("\u00a0", " ")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def normalize_block_type(value: str | None) -> str:
    """Đưa các label layout khác nhau về vocabulary block type của v2."""

    label = normalize_ocr_text(value or "").lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "plain_text": "text",
        "plaintext": "text",
        "heading": "title",
        "tablefootnote": "table_footnote",
        "figurecaption": "figure_caption",
        "discard": "abandon",
    }
    return aliases.get(label, label or "text")


def is_indexable_block(block: OCRBlock) -> bool:
    """Quyết định block có tạo semantic unit hay chỉ được giữ để audit.

    Heuristic kết hợp type, nội dung, độ dài và lexical pattern. ``abandon`` không
    bị loại mù quáng: header/cơ quan/số văn bản bị gán nhầm vẫn được giữ.
    """

    text = block.content_normalized.strip()
    if not text:
        return False
    if _PAGE_NUMBER_RE.fullmatch(text):
        return False
    if block.block_type == "table_footnote" and _SHORT_OPERATOR_NOTE_RE.fullmatch(text):
        return False
    if _STAMP_NOISE_RE.match(text):
        return False
    if block.block_type == "figure":
        return False
    if block.block_type == "abandon":
        return bool(_VALUABLE_ABANDON_RE.search(text))
    if len(text) <= 2 and not re.search(r"\w", text):
        return False
    return True


def sort_blocks_in_reading_order(blocks: Iterable[OCRBlock]) -> list[OCRBlock]:
    """Sắp block theo trang, y trên, x trái và block index làm tie-breaker.

    ``bbox`` được OCR parser chuẩn hoá về 0..1 khi biết kích thước trang. Với JSON
    thiếu kích thước, thứ tự tương đối vẫn đúng vì cùng dùng hệ pixel.
    """

    def key(block: OCRBlock) -> tuple[float, float, float, int]:
        """Tạo sort key ổn định; bbox thiếu được xếp sau trong cùng page."""

        if block.bbox is None:
            return (float(block.page_number), float("inf"), float("inf"), block.block_index)
        x1, y1, _, _ = block.bbox
        return (float(block.page_number), round(float(y1), 4), float(x1), block.block_index)

    return sorted(blocks, key=key)
