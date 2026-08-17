"""CLI inspect SAHC-v2 chunks trước khi embedding/upsert Qdrant."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .chunk_builder import build_chunks_from_ocr_document, build_document_chunks_v2
from .models import Chunk
from .ocr_parser import load_ocr_json
from .token_counter import RegexTokenizer, TokenCounter


DEFAULT_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"


def chunks_to_debug_dict(
    chunks: list[Chunk],
    document_meta: dict[str, Any],
) -> dict[str, Any]:
    """Chia output thành parents/children và thêm token statistics dễ inspect."""

    parents = [asdict(chunk) for chunk in chunks if chunk.metadata.get("record_type") == "parent"]
    children = [asdict(chunk) for chunk in chunks if chunk.metadata.get("record_type") == "child"]
    token_counts = [chunk["token_count"] for chunk in children]
    return {
        "document": document_meta,
        "statistics": {
            "parent_count": len(parents),
            "child_count": len(children),
            "average_child_tokens": statistics.fmean(token_counts) if token_counts else 0.0,
            "max_child_tokens": max(token_counts, default=0),
            "fallback_token_splits": sum(
                child["metadata"].get("split_fallback") == "token_window"
                for child in children
            ),
        },
        "parents": parents,
        "children": children,
    }

def chunks_to_markdown(chunks: list[Chunk], document_meta: dict[str, Any]) -> str:
    """Render parent/child hierarchy thành Markdown để review thủ công."""

    title = str(document_meta.get("Summary") or document_meta.get("Id") or "Document")
    children_by_parent: dict[str, list[Chunk]] = {}
    parents = [chunk for chunk in chunks if chunk.metadata.get("record_type") == "parent"]
    for chunk in chunks:
        if chunk.metadata.get("record_type") == "child" and chunk.parent_id:
            children_by_parent.setdefault(chunk.parent_id, []).append(chunk)

    lines = [f"# {title}", ""]
    for parent in parents:
        if parent.chunk_type == "table":
            continue
        label = " > ".join(parent.section_path) or parent.chunk_type
        lines.extend([f"## Parent: {label}", ""])
        for child in children_by_parent.get(parent.chunk_id, []):
            lines.extend(
                [
                    f"### Child {child.chunk_index} — {child.chunk_type}",
                    "",
                    f"- Pages: {child.page_start}-{child.page_end}",
                    f"- Tokens: {child.token_count}",
                    f"- Source blocks: {', '.join(child.source_block_ids)}",
                    "",
                    child.retrieval_text,
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def write_debug_outputs(
    chunks: list[Chunk],
    document_meta: dict[str, Any],
    json_output: str | Path,
    markdown_output: str | Path | None = None,
) -> None:
    """Ghi debug JSON bắt buộc và Markdown tùy chọn bằng UTF-8."""

    Path(json_output).write_text(
        json.dumps(chunks_to_debug_dict(chunks, document_meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if markdown_output:
        Path(markdown_output).write_text(
            chunks_to_markdown(chunks, document_meta),
            encoding="utf-8",
        )


def build_argument_parser() -> argparse.ArgumentParser:
    """Khai báo CLI arguments; offline tokenizer phải được bật explicit."""

    parser = argparse.ArgumentParser(description="Inspect Structure-Aware Chunking v2")
    parser.add_argument("--input", required=True, help="Đường dẫn OCR JSON")
    parser.add_argument("--meta", help="JSON object metadata; bỏ trống sẽ dùng input_file")
    parser.add_argument("--output", required=True, help="File chunks_debug.json")
    parser.add_argument("--output-markdown", help="File chunks_debug.md tùy chọn")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model")
    parser.add_argument(
        "--offline-tokenizer",
        action="store_true",
        help="Dùng RegexTokenizer xấp xỉ chỉ để debug không mạng",
    )
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Chạy debug pipeline bằng tokenizer thật mặc định hoặc offline mode explicit."""

    args = build_argument_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    meta = _load_meta(args.meta, args.input)
    if args.offline_tokenizer:
        document = load_ocr_json(args.input)
        counter = TokenCounter(
            tokenizer=RegexTokenizer(args.max_seq_length),
            max_seq_length=args.max_seq_length,
        )
        chunks = build_chunks_from_ocr_document(document, meta, counter)
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.model)
        chunks = build_document_chunks_v2(args.input, meta, model)
    write_debug_outputs(chunks, meta, args.output, args.output_markdown)
    return 0


def _load_meta(meta_path: str | None, input_path: str) -> dict[str, Any]:
    """Đọc metadata JSON hoặc tạo ID deterministic tối thiểu từ tên OCR input."""

    if meta_path:
        value = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("--meta phải chứa JSON object")
        return value
    return {"Id": Path(input_path).stem, "Summary": Path(input_path).stem}


if __name__ == "__main__":
    raise SystemExit(main())

