"""CLI end-to-end cho Structure-Aware Hierarchical Chunking v2.

Luồng xử lý của script:

1. Tìm một hoặc nhiều OCR JSON.
2. Ghép metadata trong ``data/documents.csv`` theo ``FileNameMinio``.
3. Chạy package ``services.chunking`` và validation token/parent/table context.
4. Xuất ``chunks.json``, ``chunks.md``, ``parents.json`` và log để debug.
5. Khi operator bật ``--upsert``: embedding child chunks và lưu vào Qdrant.

Parent chunks không được đưa vào vector database. Chúng được lưu riêng ở
``parents.json`` để retrieval layer có thể mở rộng context sau khi tìm child.

Ví dụ chỉ debug, không cần SentenceTransformer/Qdrant::

    python -m services.embedding_v2 --offline-tokenizer --verbose

Ví dụ chạy production-like và upsert Qdrant local::

    python -m services.embedding_v2 \
      --input data/file_contents \
      --upsert --create-collection --verbose
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

# Hỗ trợ cả hai cách chạy:
#   python -m services.embedding_v2
#   python services/embedding_v2.py
if __package__:
    from .chunking.chunk_builder import (
        build_chunks_from_ocr_document,
        build_document_chunks_v2,
    )
    from .chunking.debug import write_debug_outputs
    from .chunking.integration import (
        build_parent_store,
        create_v2_collection_explicit,
        embedding_children,
        upsert_document_v2,
    )
    from .chunking.models import Chunk
    from .chunking.ocr_parser import load_ocr_json
    from .chunking.token_counter import RegexTokenizer, TokenCounter
else:
    from chunking.chunk_builder import (  # type: ignore[no-redef]
        build_chunks_from_ocr_document,
        build_document_chunks_v2,
    )
    from chunking.debug import write_debug_outputs  # type: ignore[no-redef]
    from chunking.integration import (  # type: ignore[no-redef]
        build_parent_store,
        create_v2_collection_explicit,
        embedding_children,
        upsert_document_v2,
    )
    from chunking.models import Chunk  # type: ignore[no-redef]
    from chunking.ocr_parser import load_ocr_json  # type: ignore[no-redef]
    from chunking.token_counter import RegexTokenizer, TokenCounter  # type: ignore[no-redef]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "file_contents"
DEFAULT_DOCUMENTS_CSV = PROJECT_ROOT / "data" / "documents.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "embedding_v2"
DEFAULT_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://14.225.210.182:6333")
DEFAULT_COLLECTION = os.getenv("COLLECTION_NAME_V2", "rag_document_v2")
PROTECTED_V1_COLLECTION = "rag_document"

LOGGER = logging.getLogger("embedding_v2")


@dataclass(slots=True)
class DocumentRunResult:
    """Kết quả audit gọn của một OCR JSON trong một lần chạy."""

    input_file: str
    status: str
    document_id: str = ""
    summary: str = ""
    parent_count: int = 0
    child_count: int = 0
    table_row_count: int = 0
    average_child_tokens: float = 0.0
    max_child_tokens: int = 0
    fallback_token_splits: int = 0
    qdrant_points_upserted: int = 0
    qdrant_document_points: int | None = None
    debug_json: str = ""
    debug_markdown: str = ""
    parent_store: str = ""
    elapsed_seconds: float = 0.0
    error_type: str = ""
    error: str = ""


def build_argument_parser() -> argparse.ArgumentParser:
    """Khai báo CLI cho single-file, batch debug và Qdrant upsert."""

    parser = argparse.ArgumentParser(
        description=(
            "Chunk OCR JSON bằng SAHC-v2, xuất artifact debug và tùy chọn upsert Qdrant."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Một OCR JSON hoặc thư mục JSON (mặc định: data/file_contents)",
    )
    parser.add_argument(
        "--documents-csv",
        default=str(DEFAULT_DOCUMENTS_CSV),
        help="CSV metadata chứa Id/FileNameMinio",
    )
    parser.add_argument(
        "--meta",
        help="Metadata JSON override; chỉ dùng khi --input là đúng một JSON",
    )
    parser.add_argument(
        "--glob",
        default="*.json",
        help="Pattern tìm file khi --input là thư mục (mặc định: *.json)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Tìm JSON đệ quy trong các thư mục con",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Chỉ xử lý N file đầu tiên sau khi sort; hữu ích khi debug batch",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Nơi lưu chunks/parents/log/run_summary",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="SentenceTransformer dùng để đếm token và embedding",
    )
    parser.add_argument(
        "--offline-tokenizer",
        action="store_true",
        help="Dùng RegexTokenizer chỉ để debug; không thể kết hợp --upsert",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        help="Token limit cho --offline-tokenizer (mặc định: 256)",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Không tạo chunks.md",
    )
    parser.add_argument(
        "--upsert",
        action="store_true",
        help="Embedding child chunks và upsert vào Qdrant",
    )
    parser.add_argument(
        "--create-collection",
        action="store_true",
        help="Tạo collection nếu chưa tồn tại; không recreate/delete collection",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
        help="Qdrant REST URL (mặc định: QDRANT_URL hoặc 14.225.210.182:6333)",
    )
    parser.add_argument(
        "--qdrant-api-key",
        default=os.getenv("QDRANT_API_KEY"),
        help="API key; nên truyền qua QDRANT_API_KEY thay vì command history",
    )
    parser.add_argument(
        "--qdrant-timeout",
        type=float,
        default=60.0,
        help="Timeout kết nối Qdrant tính bằng giây",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Collection v2 (mặc định: COLLECTION_NAME_V2 hoặc rag_document_v2)",
    )
    parser.add_argument(
        "--log-file",
        help="File log; mặc định <output-dir>/embedding_v2.log",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Dừng batch ngay khi một tài liệu lỗi",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Hiện log DEBUG ngoài các metrics INFO bắt buộc",
    )
    return parser


def configure_logging(
    output_dir: Path,
    log_file: str | None,
    *,
    verbose: bool,
) -> Path:
    """Ghi cùng một log ra terminal và file UTF-8 để debug sau khi chạy."""

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_log = Path(log_file).expanduser() if log_file else output_dir / "embedding_v2.log"
    resolved_log.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(resolved_log, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logging.basicConfig(
        level=level,
        handlers=[console_handler, file_handler],
        force=True,
    )
    # HTTP request logs rất dài; pipeline vẫn log các mốc Qdrant quan trọng ở INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return resolved_log.resolve()


def discover_json_files(
    input_path: str | Path,
    *,
    pattern: str = "*.json",
    recursive: bool = False,
    limit: int | None = None,
) -> list[Path]:
    """Resolve input thành danh sách OCR JSON có thứ tự deterministic."""

    path = Path(input_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Không tồn tại --input: {path}")
    if limit is not None and limit <= 0:
        raise ValueError("--limit phải lớn hơn 0")
    if path.is_file():
        if path.suffix.casefold() != ".json":
            raise ValueError(f"--input file phải có đuôi .json: {path}")
        files = [path]
    else:
        iterator = path.rglob(pattern) if recursive else path.glob(pattern)
        files = sorted(
            (item for item in iterator if item.is_file() and item.suffix.casefold() == ".json"),
            key=lambda item: str(item).casefold(),
        )
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"Không tìm thấy OCR JSON với pattern {pattern!r} trong {path}")
    return [item.resolve() for item in files]


def _meta_value(metadata: dict[str, Any], *field_names: str) -> str:
    """Lấy field metadata không phân biệt hoa/thường và chuẩn hóa thành string."""

    by_lower = {str(key).casefold(): value for key, value in metadata.items()}
    for field_name in field_names:
        value = by_lower.get(field_name.casefold())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def load_metadata_index(csv_path: str | Path) -> dict[str, dict[str, str]]:
    """Đọc documents.csv và index row theo stem của ``FileNameMinio``.

    Tên JSON thật đang có dạng ``{FileIdMinio}_{Page}.json``; stem của
    ``FileNameMinio`` trong CSV có đúng quy ước này. Hàm cũng thêm alias từ
    ``FileIdMinio`` và ``Page`` để tolerant CSV thiếu tên PDF.
    """

    path = Path(csv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Không tồn tại metadata CSV: {path}")
    index: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV không có header: {path}")
        for row_number, row in enumerate(reader, start=2):
            normalized_row = {str(key): value or "" for key, value in row.items() if key}
            document_id = _meta_value(normalized_row, "Id")
            if not document_id:
                LOGGER.warning("Bỏ qua CSV row %d vì thiếu Id", row_number)
                continue
            aliases: set[str] = set()
            file_name = _meta_value(normalized_row, "FileNameMinio")
            if file_name:
                aliases.add(Path(file_name).stem.casefold())
            file_id = _meta_value(normalized_row, "FileIdMinio")
            page = _meta_value(normalized_row, "Page")
            if file_id:
                aliases.add(file_id.casefold())
                if page:
                    aliases.add(f"{file_id}_{page}".casefold())
            for alias in aliases:
                existing = index.get(alias)
                if existing and _meta_value(existing, "Id") != document_id:
                    raise ValueError(
                        f"Metadata alias bị trùng {alias!r} ở CSV row {row_number}"
                    )
                index[alias] = normalized_row
    if not index:
        raise ValueError(f"Không tạo được metadata index từ {path}")
    return index


def load_metadata_override(meta_path: str | Path) -> dict[str, Any]:
    """Đọc ``--meta`` JSON object cho một tài liệu đơn lẻ."""

    path = Path(meta_path).expanduser()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("--meta phải chứa một JSON object")
    if not _meta_value(value, "Id"):
        raise ValueError("--meta thiếu Id; không thể sinh chunk ID ổn định")
    return value


def resolve_document_metadata(
    json_path: Path,
    metadata_index: dict[str, dict[str, str]],
    override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ghép OCR JSON với metadata và từ chối fallback ID không ổn định."""

    if override is not None:
        return dict(override)
    metadata = metadata_index.get(json_path.stem.casefold())
    if metadata is None:
        raise KeyError(
            f"Không tìm thấy metadata cho {json_path.name}. "
            "Kiểm tra FileNameMinio/FileIdMinio/Page trong documents.csv hoặc dùng --meta."
        )
    return dict(metadata)


def load_embedding_model(model_name: str) -> Any:
    """Lazy-load SentenceTransformer để offline debug không cần dependency này."""

    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Thiếu sentence-transformers. Chạy: "
            "python -m pip install -r services/chunking/requirements.txt"
        ) from exc
    LOGGER.info("[model] loading=%s", model_name)
    model = SentenceTransformer(model_name)
    dimension_method = getattr(model, "get_sentence_embedding_dimension", None)
    dimension = dimension_method() if callable(dimension_method) else "unknown"
    LOGGER.info(
        "[model] ready=%s dimension=%s max_seq_length=%s",
        model_name,
        dimension,
        getattr(model, "max_seq_length", "unknown"),
    )
    return model


def create_qdrant_client(url: str, api_key: str | None, timeout: float) -> Any:
    """Lazy-load QdrantClient và tạo connection config không làm lộ API key."""

    try:
        from qdrant_client import QdrantClient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Thiếu qdrant-client. Chạy: "
            "python -m pip install -r services/chunking/requirements.txt"
        ) from exc
    kwargs: dict[str, Any] = {"url": url, "timeout": timeout}
    if api_key:
        kwargs["api_key"] = api_key
    return QdrantClient(**kwargs)


def _existing_collection_vector_size(client: Any, collection_name: str) -> int | None:
    """Đọc vector dimension của collection unnamed-vector nếu client hỗ trợ."""

    info = client.get_collection(collection_name=collection_name)
    config = getattr(info, "config", None)
    params = getattr(config, "params", None)
    vectors = getattr(params, "vectors", None)
    size = getattr(vectors, "size", None)
    if size is not None:
        return int(size)
    if isinstance(vectors, dict):
        # Pipeline build PointStruct với unnamed vector nên named vectors không tương thích.
        raise ValueError(
            f"Collection {collection_name!r} dùng named vectors; embedding_v2 cần unnamed vector"
        )
    return None


def ensure_qdrant_collection(
    client: Any,
    embedding_model: Any,
    collection_name: str,
    *,
    create_if_missing: bool,
) -> None:
    """Kiểm tra collection/dimension; chỉ tạo mới khi operator cho phép rõ ràng."""

    if collection_name == PROTECTED_V1_COLLECTION:
        raise ValueError(
            "Từ chối ghi collection v1 'rag_document'; hãy dùng rag_document_v2 hoặc tên test"
        )
    if client.collection_exists(collection_name=collection_name):
        expected_method = getattr(embedding_model, "get_sentence_embedding_dimension", None)
        expected = int(expected_method()) if callable(expected_method) else None
        actual = _existing_collection_vector_size(client, collection_name)
        if expected is not None and actual is not None and expected != actual:
            raise ValueError(
                f"Collection {collection_name!r} dimension={actual}, model dimension={expected}. "
                "Hãy dùng collection v2 khác; script không tự recreate collection."
            )
        LOGGER.info(
            "[qdrant] collection_ready name=%s dimension=%s",
            collection_name,
            actual if actual is not None else "unknown",
        )
        return
    if not create_if_missing:
        raise RuntimeError(
            f"Collection {collection_name!r} chưa tồn tại. "
            "Chạy lại với --create-collection để tạo explicit."
        )
    LOGGER.info("[qdrant] creating_collection name=%s", collection_name)
    create_v2_collection_explicit(
        client,
        embedding_model,
        collection_name=collection_name,
    )
    LOGGER.info("[qdrant] collection_created name=%s", collection_name)


def count_qdrant_document_points(
    client: Any,
    collection_name: str,
    document_id: str,
) -> int:
    """Đếm chính xác child points của một document sau upsert để audit."""

    from qdrant_client.models import FieldCondition, Filter, MatchValue

    result = client.count(
        collection_name=collection_name,
        count_filter=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        ),
        exact=True,
    )
    return int(result.count)


def _chunk_statistics(chunks: Sequence[Chunk]) -> dict[str, int | float]:
    """Tổng hợp metrics child/parent từ output đã qua validation."""

    parents = [chunk for chunk in chunks if chunk.metadata.get("record_type") == "parent"]
    children = embedding_children(list(chunks))
    tokens = [chunk.token_count for chunk in children]
    return {
        "parent_count": len(parents),
        "child_count": len(children),
        "table_row_count": sum(chunk.chunk_type == "table_row" for chunk in children),
        "average_child_tokens": statistics.fmean(tokens) if tokens else 0.0,
        "max_child_tokens": max(tokens, default=0),
        "fallback_token_splits": sum(
            chunk.metadata.get("split_fallback") == "token_window" for chunk in children
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    """Ghi JSON UTF-8 có indent và tự tạo thư mục cha."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def process_document(
    json_path: Path,
    metadata: dict[str, Any],
    *,
    output_dir: Path,
    embedding_model: Any | None,
    offline_counter: TokenCounter | None,
    write_markdown: bool,
    qdrant_client: Any | None,
    collection_name: str,
    upsert: bool,
) -> DocumentRunResult:
    """Chạy đủ parse → chunk → validate → artifact → optional Qdrant cho một file."""

    started = time.perf_counter()
    document_id = _meta_value(metadata, "Id")
    summary = _meta_value(metadata, "Summary")
    document_output = output_dir / json_path.stem
    document_output.mkdir(parents=True, exist_ok=True)
    debug_json = document_output / "chunks.json"
    debug_markdown = document_output / "chunks.md"
    parent_path = document_output / "parents.json"

    LOGGER.info(
        "[document] start file=%s document_id=%s summary=%s",
        json_path,
        document_id,
        summary,
    )
    if offline_counter is not None:
        LOGGER.debug("[document] stage=parse_ocr tokenizer=regex")
        ocr_document = load_ocr_json(json_path)
        chunks = build_chunks_from_ocr_document(
            ocr_document,
            document_meta=metadata,
            token_counter=offline_counter,
        )
    else:
        if embedding_model is None:
            raise RuntimeError("Thiếu embedding model cho production-like chunking")
        LOGGER.debug("[document] stage=parse_chunk_validate tokenizer=model")
        chunks = build_document_chunks_v2(
            json_path=str(json_path),
            document_meta=metadata,
            embedding_model=embedding_model,
        )

    metrics = _chunk_statistics(chunks)
    LOGGER.info(
        "[document] chunked document_id=%s parents=%d children=%d table_rows=%d "
        "max_tokens=%d avg_tokens=%.1f fallback_token_splits=%d",
        document_id,
        metrics["parent_count"],
        metrics["child_count"],
        metrics["table_row_count"],
        metrics["max_child_tokens"],
        metrics["average_child_tokens"],
        metrics["fallback_token_splits"],
    )

    write_debug_outputs(
        chunks,
        metadata,
        json_output=debug_json,
        markdown_output=debug_markdown if write_markdown else None,
    )
    parent_store = build_parent_store(chunks, metadata)
    _write_json(parent_path, parent_store)
    LOGGER.info(
        "[document] artifacts chunks=%s markdown=%s parents=%s",
        debug_json,
        debug_markdown if write_markdown else "disabled",
        parent_path,
    )

    inserted = 0
    qdrant_document_points: int | None = None
    if upsert:
        if qdrant_client is None or embedding_model is None:
            raise RuntimeError("--upsert cần Qdrant client và embedding model")
        LOGGER.info(
            "[qdrant] embedding_and_upsert document_id=%s children=%d collection=%s",
            document_id,
            metrics["child_count"],
            collection_name,
        )
        inserted = upsert_document_v2(
            qdrant_client,
            chunks,
            metadata,
            embedding_model,
            collection_name=collection_name,
        )
        qdrant_document_points = count_qdrant_document_points(
            qdrant_client,
            collection_name,
            document_id,
        )
        LOGGER.info(
            "[qdrant] upserted document_id=%s points=%d stored_for_document=%d",
            document_id,
            inserted,
            qdrant_document_points,
        )
        if qdrant_document_points != inserted:
            LOGGER.warning(
                "[qdrant] document_point_count_mismatch document_id=%s current=%d "
                "this_run=%d; upsert không xóa point cũ. Hãy dùng collection mới "
                "hoặc xóa theo document_id bằng thao tác explicit sau khi review.",
                document_id,
                qdrant_document_points,
                inserted,
            )

    elapsed = time.perf_counter() - started
    LOGGER.info("[document] success document_id=%s elapsed=%.2fs", document_id, elapsed)
    return DocumentRunResult(
        input_file=str(json_path),
        status="PASS",
        document_id=document_id,
        summary=summary,
        parent_count=int(metrics["parent_count"]),
        child_count=int(metrics["child_count"]),
        table_row_count=int(metrics["table_row_count"]),
        average_child_tokens=round(float(metrics["average_child_tokens"]), 2),
        max_child_tokens=int(metrics["max_child_tokens"]),
        fallback_token_splits=int(metrics["fallback_token_splits"]),
        qdrant_points_upserted=inserted,
        qdrant_document_points=qdrant_document_points,
        debug_json=str(debug_json.resolve()),
        debug_markdown=str(debug_markdown.resolve()) if write_markdown else "",
        parent_store=str(parent_path.resolve()),
        elapsed_seconds=round(elapsed, 3),
    )


def write_run_summary(
    output_dir: Path,
    results: Sequence[DocumentRunResult],
    *,
    args: argparse.Namespace,
    log_path: Path,
) -> Path:
    """Ghi manifest cuối run, tuyệt đối không serialize Qdrant API key."""

    passed = sum(result.status == "PASS" for result in results)
    failed = len(results) - passed
    payload = {
        "configuration": {
            "input": str(Path(args.input).expanduser()),
            "documents_csv": str(Path(args.documents_csv).expanduser()),
            "model": "RegexTokenizer" if args.offline_tokenizer else args.model,
            "max_seq_length": args.max_seq_length if args.offline_tokenizer else "model",
            "upsert": args.upsert,
            "qdrant_url": args.qdrant_url if args.upsert else None,
            "collection": args.collection if args.upsert else None,
            "log_file": str(log_path),
        },
        "statistics": {
            "documents": len(results),
            "passed": passed,
            "failed": failed,
            "parents": sum(result.parent_count for result in results),
            "children": sum(result.child_count for result in results),
            "qdrant_points_upserted": sum(
                result.qdrant_points_upserted for result in results
            ),
        },
        "documents": [asdict(result) for result in results],
    }
    summary_path = output_dir / "run_summary.json"
    _write_json(summary_path, payload)
    return summary_path.resolve()


def _validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Chặn các tổ hợp option không an toàn trước khi load model hoặc kết nối DB."""

    if args.offline_tokenizer and args.upsert:
        parser.error("--offline-tokenizer chỉ dùng debug, không thể kết hợp --upsert")
    if args.create_collection and not args.upsert:
        parser.error("--create-collection chỉ có ý nghĩa khi đi cùng --upsert")
    if args.max_seq_length <= 0:
        parser.error("--max-seq-length phải lớn hơn 0")
    if args.qdrant_timeout <= 0:
        parser.error("--qdrant-timeout phải lớn hơn 0")
    if args.upsert and args.collection == PROTECTED_V1_COLLECTION:
        parser.error("Từ chối --collection rag_document; hãy dùng collection v2")


def main(argv: Sequence[str] | None = None) -> int:
    """Điều phối batch, giữ lỗi theo từng document và trả exit code phù hợp automation."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    log_path = configure_logging(output_dir, args.log_file, verbose=args.verbose)
    LOGGER.info(
        "[run] start input=%s output=%s mode=%s upsert=%s collection=%s log=%s",
        args.input,
        output_dir,
        "offline-debug" if args.offline_tokenizer else "model-tokenizer",
        args.upsert,
        args.collection if args.upsert else "disabled",
        log_path,
    )

    results: list[DocumentRunResult] = []
    try:
        json_files = discover_json_files(
            args.input,
            pattern=args.glob,
            recursive=args.recursive,
            limit=args.limit,
        )
        if args.meta and len(json_files) != 1:
            raise ValueError("--meta chỉ được dùng khi --input resolve đúng một JSON")
        metadata_override = load_metadata_override(args.meta) if args.meta else None
        metadata_index = (
            {} if metadata_override is not None else load_metadata_index(args.documents_csv)
        )
        LOGGER.info("[run] discovered_json=%d", len(json_files))

        embedding_model: Any | None = None
        offline_counter: TokenCounter | None = None
        if args.offline_tokenizer:
            offline_counter = TokenCounter(
                tokenizer=RegexTokenizer(args.max_seq_length),
                max_seq_length=args.max_seq_length,
            )
            LOGGER.info(
                "[model] using=RegexTokenizer max_seq_length=%d debug_only=true",
                args.max_seq_length,
            )
        else:
            embedding_model = load_embedding_model(args.model)

        qdrant_client: Any | None = None
        if args.upsert:
            LOGGER.info("[qdrant] connecting url=%s", args.qdrant_url)
            qdrant_client = create_qdrant_client(
                args.qdrant_url,
                args.qdrant_api_key,
                args.qdrant_timeout,
            )
            ensure_qdrant_collection(
                qdrant_client,
                embedding_model,
                args.collection,
                create_if_missing=args.create_collection,
            )

        for json_path in json_files:
            started = time.perf_counter()
            try:
                metadata = resolve_document_metadata(
                    json_path,
                    metadata_index,
                    metadata_override,
                )
                result = process_document(
                    json_path,
                    metadata,
                    output_dir=output_dir,
                    embedding_model=embedding_model,
                    offline_counter=offline_counter,
                    write_markdown=not args.no_markdown,
                    qdrant_client=qdrant_client,
                    collection_name=args.collection,
                    upsert=args.upsert,
                )
            except Exception as exc:  # Giữ batch chạy tiếp và lưu đầy đủ traceback trong log.
                elapsed = time.perf_counter() - started
                LOGGER.exception("[document] failed file=%s", json_path)
                result = DocumentRunResult(
                    input_file=str(json_path),
                    status="FAIL",
                    elapsed_seconds=round(elapsed, 3),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            results.append(result)
            if result.status == "FAIL" and args.fail_fast:
                LOGGER.error("[run] fail_fast=true; stopping batch")
                break
    except Exception as exc:
        LOGGER.exception("[run] fatal_error")
        results.append(
            DocumentRunResult(
                input_file=str(args.input),
                status="FAIL",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        )

    summary_path = write_run_summary(output_dir, results, args=args, log_path=log_path)
    passed = sum(result.status == "PASS" for result in results)
    failed = len(results) - passed
    LOGGER.info(
        "[run] complete documents=%d passed=%d failed=%d summary=%s",
        len(results),
        passed,
        failed,
        summary_path,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
