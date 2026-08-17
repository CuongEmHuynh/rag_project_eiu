# SAHC-v2 — Structure-Aware Hierarchical Chunking

Implementation trong thư mục này bám theo `CHUNKING_V2_CODEX_SPEC.md`:

```text
OCR JSON
→ normalize/filter/sort layout blocks
→ parse cây cấu trúc tài liệu
→ parse + ghép bảng qua trang
→ tạo semantic atomic units
→ prefix-aware token packing
→ tạo parent/child chunks
→ build retrieval text
→ validation
→ child embedding / parent store
```

Không có raw HTML table nào được dùng làm embedding child chính. Mỗi table row
được serialize với schema; bảng chuyển môn phân biệt rõ môn đã học và môn được
chuyển. Parent không dùng zero vector và không được đưa vào child vector search.

## Cấu trúc source

| File | Trách nhiệm |
|---|---|
| `models.py` | Dataclass contract cho OCR block, tree, table, atomic unit, packed unit, chunk và config. |
| `normalize.py` | Unicode/whitespace normalization, block type alias, noise filtering và reading order. |
| `ocr_parser.py` | Đọc OCR JSON chuẩn hoặc alias field thường gặp, giữ song song raw/normalized text. |
| `structure_parser.py` | Parse document/preamble/legal basis/Điều/Khoản/Điểm/table/Nơi nhận/chữ ký. |
| `table_parser.py` | Parse HTML, rowspan/colspan, multi-row header, course serializer, continuation scoring/merge. |
| `token_counter.py` | Adapter tokenizer thật; `RegexTokenizer` chỉ dành cho test/debug offline explicit. |
| `token_packer.py` | Packing trên retrieval text đầy đủ; semantic split trước token-window fallback. |
| `chunk_builder.py` | Orchestrator tạo parent/table-parent/embedding-child với UUID5 ổn định. |
| `retrieval_text.py` | Metadata prefix, null-safe retrieval text và payload v2. |
| `validators.py` | Chặn missing parent, empty chunk, table context thiếu và token overflow. |
| `integration.py` | Feature flag, OCR path, v1 fallback, parent store, Qdrant points/upsert an toàn. |
| `retrieval.py` | Child-only search, per-parent dedup và context expansion. |
| `legacy.py` | Baseline `chunk_legal_document_v1()` để A/B test/rollback. |
| `debug.py` | CLI xuất JSON và Markdown để review chunk trước khi index. |
| `tests/` | 18 tests structure/table/token/parent-child/retrieval/fallback/real tokenizer. |

Chi tiết ý nghĩa của từng function nằm trong [FUNCTION_REFERENCE.md](FUNCTION_REFERENCE.md).
Ngoài ra, mọi function/class trong source đều có docstring tiếng Việt ngay tại code.

## API chính

```python
from sentence_transformers import SentenceTransformer

from services.chunking import build_document_chunks_v2
from services.chunking.integration import build_parent_store, build_qdrant_points

model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")
meta = {
    "Id": "document-uuid",
    "Summary": "Quyết định chuyển điểm cho sinh viên ...",
    "No": "01/QĐ-EIU",
    "Author": "Trường Đại học Quốc tế Miền Đông",
    "DateDocument": "01/01/2016",
}

chunks = build_document_chunks_v2(
    json_path="data/file_contents/document-uuid.json",
    document_meta=meta,
    embedding_model=model,
)

# Parent lưu JSON/DB/cache riêng; chỉ child mới được encode thành Qdrant points.
parent_store = build_parent_store(chunks, meta)
points = build_qdrant_points(chunks, meta, model)
```

`build_document_chunks_v2()` trả chung parent và child để validation có thể kiểm
tra quan hệ. Lọc child bằng `integration.embedding_children()` hoặc field:

```python
child_chunks = [
    chunk for chunk in chunks
    if chunk.metadata["record_type"] == "child"
]
```

## Feature flag và fallback

```bash
export CHUNKING_VERSION=v2
export OCR_JSON_DIR=./data/file_contents
export COLLECTION_NAME_V2=rag_document_v2
```

```python
from services.chunking.integration import load_chunks_by_version

chunks = load_chunks_by_version(meta, model)
```

- `v2`: ưu tiên `{Id}.json`.
- Nếu JSON thiếu và `ChunkingConfig.enable_v1_txt_fallback=True`, TXT fallback
  được dùng với warning và payload `chunking_version=v1-fallback`,
  `source=OCR_TXT_FALLBACK`.
- `v1`: baseline được chọn explicit.
- `create_v2_collection_explicit()` từ chối tên `rag_document` và không được gọi
  tự động. `upsert_document_v2()` chỉ upsert vào collection đã tồn tại.

## Debug CLI

Tokenizer thật (production-like):

```bash
PYTHONPATH=services python -m chunking.debug \
  --input data/file_contents/<Id>.json \
  --meta path/to/meta.json \
  --output /tmp/chunks_debug.json \
  --output-markdown /tmp/chunks_debug.md
```

Debug offline không tải model (token count chỉ xấp xỉ, phải bật explicit):

```bash
PYTHONPATH=services python -m chunking.debug \
  --input path/to/file.json \
  --output /tmp/chunks_debug.json \
  --output-markdown /tmp/chunks_debug.md \
  --offline-tokenizer --max-seq-length 256
```

JSON output có ba key chính: `document`, `parents`, `children`; Markdown nhóm
children theo Article/Preamble/Recipients/Signature parent.

## Qdrant payload child

```json
{
  "document_id": "...",
  "chunk_id": "UUID5 deterministic",
  "chunk_index": 7,
  "chunk_type": "table_row",
  "record_type": "child",
  "parent_id": "article-parent-uuid",
  "table_parent_id": "logical-table-parent-uuid",
  "table_id": "logical-table-id",
  "table_row_index": 3,
  "section_path": ["QUYẾT ĐỊNH", "Điều 1", "Bảng"],
  "page_start": 2,
  "page_end": 2,
  "raw_text": "PHYS 201 | Vật lý 1A | ...",
  "normalized_text": "Môn học đã học: ...",
  "retrieval_text": "Văn bản: ...\nPhần: ...\n\nNội dung:\n...",
  "token_count": 137,
  "source_block_ids": ["page_002_block_0001"],
  "source": "OCR_JSON",
  "chunking_version": "v2"
}
```

## Test

Cài dependency production khi cần:

```bash
python -m pip install -r services/chunking/requirements.txt
```

Chạy toàn bộ test (không cần pytest):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m unittest discover -s services/chunking/tests -v
```

Test tokenizer thật tự dùng snapshot local nếu có; nếu không có thì skip và các
unit test vẫn dùng dependency injection như đặc tả cho phép.

## Hạn chế hiện tại

- Workspace hiện không có ba OCR JSON production được nêu trong spec trong
  `data/file_contents/`; tests dùng fixtures tổng hợp tương ứng Hồ Xuân Tường,
  Phạm Minh Quân và Võ Hoàng Duy. Cần chạy debug CLI trên JSON thật trước migration.
- Structure parser v2 là deterministic heuristic, chưa giải layout nhiều cột tổng
  quát hoặc OCR correction (đúng non-goal của spec).
- Course-transfer serializer hỗ trợ schema 8 cột và biến thể có cột STT; schema lạ
  fallback an toàn về `Header: value`, không hallucinate ô thiếu.
- Parent store được trả dưới dạng dictionary; ứng dụng cần chọn JSON/DB/cache bền
  vững phù hợp hạ tầng triển khai.
- Root virtualenv tại thời điểm triển khai thiếu `sentence-transformers` và
  `qdrant-client`; hai dependency này được khai báo trong requirements cục bộ trên.

