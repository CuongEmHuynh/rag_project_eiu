"""Baseline V1 được giữ nguyên tinh thần để A/B test và rollback explicit."""

from __future__ import annotations

import re


def clean_data_v1(text: str) -> str:
    """Giữ logic clean baseline cũ; không dùng cho raw OCR của pipeline v2."""

    text = text.replace("\u00a0", " ")
    text = re.sub(r"[?]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_legal_document_v1(text: str) -> list[tuple[str, str]]:
    """Baseline split ``Điều x:`` + char threshold để benchmark với SAHC-v2.

    Hàm cố ý giữ character-based behavior cũ; production v2 không gọi logic này
    trừ khi feature flag/fallback được bật explicit.
    """

    chunks: list[tuple[str, str]] = []
    for part in re.split(r"(?=Điều\s+\d+\s*:)", text):
        part = part.strip()
        if not part:
            continue
        chunk_type = "dieu" if part.startswith("Điều") else "header"
        if len(part) > 3000:
            for sub_part in re.split(r"\n{2,}", part):
                if len(sub_part.strip()) > 200:
                    chunks.append((chunk_type, sub_part.strip()))
        else:
            chunks.append((chunk_type, part))
    return chunks


# Backward-compatible alias cho code cũ import tên này.
chunk_legal_document = chunk_legal_document_v1

