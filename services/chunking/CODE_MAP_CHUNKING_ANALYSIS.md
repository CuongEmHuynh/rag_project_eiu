# Code Map Analysis - services/chunking

Ngay 2026-08-17, da quet toan bo `services/chunking` bang `rg` va AST tinh.
Khong the chay pipeline `graphify` day du vi package `graphifyy` khong co san,
lenh cai qua PyPI bi sandbox chan mang, va yeu cau escalated install bi tu choi
vi rui ro cai package moi vao runtime bundled. Bao cao nay la code map fallback
khong dung dependency moi.

## Corpus

- 28 file trong `services/chunking`.
- 15 Python module/test file, 3 Markdown/spec/README file, 1 requirements file.
- Test suite hien bi chan o import `bs4`/`beautifulsoup4` trong runtime Codex
  bundled, du `services/chunking/requirements.txt` da khai bao dependency nay.

## Module Map

| File | Vai tro |
| --- | --- |
| `models.py` | Contract du lieu: `OCRBlock`, `OCRDocument`, `DocumentNode`, `DocumentTree`, `ParsedTable`, `AtomicUnit`, `PackedUnit`, `Chunk`, `ChunkingConfig`, `RetrievalResult`. |
| `normalize.py` | Normalize nhe OCR text, alias block type, loc noise, sort reading order. |
| `ocr_parser.py` | Doc/parse OCR JSON thanh `OCRDocument`; ho tro alias schema va bbox normalize. |
| `structure_parser.py` | State machine tao cay document/preamble/decision/article/clause/point/table/recipients/signature. |
| `table_parser.py` | Parse HTML table, flatten multi-row header, detect/merge cross-page tables, serialize table rows. |
| `token_counter.py` | Adapter tokenizer that cua embedding model; `RegexTokenizer` chi test/debug offline. |
| `token_packer.py` | Prefix-aware token packing tu `AtomicUnit` sang `PackedUnit`; token-window chi la fallback cuoi. |
| `chunk_builder.py` | Orchestrator SAHC-v2: tree + tables + atomic units + parent/child chunks + validate. |
| `retrieval_text.py` | Tao contextual retrieval text va payload v2 cho Qdrant/storage. |
| `validators.py` | Integrity gate: duplicate ID, missing parent, empty/short child, token overflow, table metadata. |
| `integration.py` | Feature flag v1/v2, OCR path resolve, v1 fallback, parent store, Qdrant point/upsert helpers. |
| `retrieval.py` | Child-only vector search, per-parent dedup, parent/table/sibling context expansion. |
| `legacy.py` | Baseline v1 regex `Dieu` + character threshold de A/B/rollback. |
| `debug.py` | CLI xuat debug JSON/Markdown de review chunk truoc khi embedding/upsert. |
| `tests/` | Acceptance tests cho structure, table, token packing, builder, integration, retrieval. |

## Call Graph Chinh

```text
build_document_chunks(json_path, meta, embedding_model)
  -> load_ocr_json()
  -> TokenCounter(embedding_model)
  -> build_chunks_from_ocr_document()

build_chunks_from_ocr_document(ocr_document, meta, token_counter)
  -> is_indexable_block()
  -> sort_blocks_in_reading_order()
  -> build_document_tree()
  -> parse_document_tables()
  -> reconstruct_cross_page_tables()
  -> create_atomic_units()
  -> _build_parent_chunks()
  -> pack_atomic_units(candidate_builder=build_retrieval_text)
  -> _build_child_chunks()
  -> _link_parent_children()
  -> validate_chunks()
  -> _log_document_stats()
```

Luot pipeline quan trong nam tai `chunk_builder.py:72-139`. Day la diem hop
nhat cua toan bo phuong phap chunking, khong phai module phu.

## Phuong Phap Chunking

### 1. Input chinh la OCR JSON

`build_document_chunks()` doc JSON bang `load_ocr_json()` va bat buoc co
embedding model de resolve tokenizer (`chunk_builder.py:43-58`). `ocr_parser.py`
giu song song `content_raw` va `content_normalized`, parse pages/blocks, alias
field OCR, va normalize bbox ve 0..1 khi co width/height (`ocr_parser.py:30-100`).

Tac dung: chunker khong con bi mat layout nhu TXT v1. Provenance duoc giu bang
`OCRBlock.block_id` dang `page_001_block_0001` (`models.py:26-30`).

### 2. Normalize va filter noise truoc khi parse

`normalize_ocr_text()` chi NFC, thay NBSP, gon whitespace/newline; khong sua loi
OCR doan noi dung phap ly (`normalize.py:23-37`). `is_indexable_block()` loai
page number, operator note, stamp noise, figure noise, nhung khong loai mu quang
`abandon` neu no co dau hieu gia tri nhu co quan, so van ban, quyet dinh, can cu,
hoac Dieu (`normalize.py:55-77`).

Tac dung: giam rac embedding nhung van giu cac block bi OCR gan sai label ma co
gia tri semantic.

### 3. Parse cau truc truoc khi dong goi token

`structure_parser.py` dung regex anchored o dau paragraph cho `Dieu`, `Khoan`,
`Diem`, `Can cu`, `Noi nhan`, `Chu ky` (`structure_parser.py:13-68`). State
machine trong `build_document_tree()` gan moi block sau mot Dieu vao Dieu hien
tai cho toi khi gap Dieu moi, `Noi nhan`, hoac signature (`structure_parser.py:71-295`).

Tac dung: day la Structure-Aware Hierarchical Chunking. Chunk child se biet
`section_path`, `parent_id`, `page_start/page_end`, va `source_block_ids`, thay vi
chi la mot doan text cat bang regex.

### 4. Table-aware row chunking

`table_parser.py` parse HTML bang BeautifulSoup, mo rong `rowspan/colspan`, nhan
header tu `thead` hoac cac row dau toan `th`, flatten multi-level header thanh
schema cot (`table_parser.py:75-131`, `table_parser.py:331-384`). Neu schema giong
bang chuyen mon, `CourseTransferTableSerializer` tach ro "Mon hoc da hoc" va
"Mon hoc duoc chuyen" (`table_parser.py:52-72`, `table_parser.py:401-415`).

`create_atomic_units()` tao moi table row thanh `AtomicUnit(unit_type="table_row")`
voi metadata `table_id`, `table_schema`, `table_row_index`, `cross_page_table`
(`chunk_builder.py:187-227`).

Tac dung: khong embed raw HTML table nhu mot blob lon. Retrieval tim tung row
chinh xac hon, nhat la cau hoi kieu "mon X duoc chuyen thanh mon nao".

### 5. Cross-page table reconstruction

`score_table_continuation()` cham diem bang trang ke nhau, vi tri gan cuoi/dau
trang, so cot, cung structural parent, khong co boundary moi truoc table, va
schema similarity (`table_parser.py:175-238`). `reconstruct_cross_page_tables()`
quet table theo thu tu va merge neu dat threshold (`table_parser.py:276-300`).

Tac dung: row o trang 2 co the ke thua header/schema cua table trang 1, tranh
mat ngu nghia cot.

### 6. Parent/child chunk design

`_build_parent_chunks()` tao parent structural theo cac parent co atomic unit, va
tao secondary table parent rieng cho logical table (`chunk_builder.py:231-326`).
`_build_child_chunks()` tao child embedding chunks, gan `parent_id`, va neu la
table row thi gan `table_parent_id` (`chunk_builder.py:329-381`).

Tac dung: child dung cho vector search; parent/table parent dung de expand
context. Parent khong bi encode bang zero-vector.

### 7. Prefix-aware token packing

`pack_atomic_units()` tinh budget tu `token_counter.max_seq_length -
safety_margin - special_token_margin`, va dem tren full retrieval text thong qua
`candidate_text_builder` (`token_packer.py:22-76`, `chunk_builder.py:106-117`).
No khong pack chung unit khac parent, khac type, khac section path, hoac khac
table (`token_packer.py:182-190`). Table row mac dinh force single row/chunk
(`token_packer.py:56-72`).

Neu mot unit qua dai, `split_oversized_atomic_unit()` thu paragraph/sentence
truoc; token-window co overlap chi dung cuoi cung va log warning
(`token_packer.py:79-163`).

Tac dung: khong con logic `len(text) > 3000` cua v1 trong core v2. Chunk child
duoc validate de khong silent truncate.

### 8. Retrieval text va payload

`build_retrieval_text()` them metadata context (`Van ban`, `So`, `Co quan ban
hanh`, `Ngay`, `Phan`) roi moi den `Noi dung` (`retrieval_text.py:18-41`).
`chunk_to_payload()` xuat day du `document_id`, `chunk_id`, `chunk_type`,
`parent_id`, `section_path`, `page_start/page_end`, raw/normalized/retrieval text,
token count, source block IDs, source, va `chunking_version=v2`
(`retrieval_text.py:44-65`).

### 9. Validation gate

`validate_chunks()` chi ap token overflow cho child embedding, kiem duplicate ID,
parent hop le, retrieval text rong/qua ngan, token_count khop thuc te, va table
row co `table_id` + `section_path` (`validators.py:22-80`).

Tac dung: loi chunking bi fail som truoc khi embedding/upsert.

### 10. Integration va retrieval v2

`integration.py` route bang `CHUNKING_VERSION`, uu tien v2 JSON, chi fallback TXT
neu config cho phep va payload ghi ro `v1-fallback`/`OCR_TXT_FALLBACK`
(`integration.py:83-117`, `integration.py:120-238`). `build_qdrant_points()` chi
encode `embedding_children()` (`integration.py:241-286`). `upsert_document_v2()`
khong tao/xoa/recreate collection (`integration.py:289-302`).

`embedding_search_v2()` filter `record_type=child`, query top-k mo rong, dedup
theo parent, roi adaptive expansion (`retrieval.py:82-111`). Với table row,
`expand_context()` uu tien table parent; prose thi lay structural parent
(`retrieval.py:33-79`).

## V1 vs V2

V1 trong `legacy.py` van la baseline: clean text, split bang regex lookahead
`(?=Dieu\s+\d+\s*:)`, chunk lon dua tren character length > 3000
(`legacy.py:8-37`). V2 thay doi representation: OCR JSON -> tree/table/atomic
units -> token-aware parent/child chunks -> child vector search + context expansion.

## Acceptance Signals Tu Tests

- Structure: test doi hoi detect dung Dieu 1-4 va khong split substring "o dieu 1"
  (`tests/test_structure_parser.py:25-29`).
- Preamble: legal basis phai nam trong preamble, khong thuoc Dieu 1
  (`tests/test_structure_parser.py:31-36`).
- Recipients: `Noi nhan` dong article cuoi va ve root-level section
  (`tests/test_structure_parser.py:38-43`).
- Noise: page number/operator footnote khong thanh semantic node
  (`tests/test_structure_parser.py:45-50`).
- Table: header da tang duoc flatten, schema course transfer duoc detect
  (`tests/test_table_parser.py:22-31`).
- Table row semantic: row co ca source/target labels va gia tri chinh
  (`tests/test_table_parser.py:33-47`).
- Cross-page table: hai physical table thanh mot logical table page 1-2
  (`tests/test_table_parser.py:49-69`).
- Builder: children co parent hop le, khong overflow, deterministic UUID, payload
  du truong v2 (`tests/test_chunk_builder.py:37-108`).
- Retrieval: dedup theo parent va adaptive table row expansion dung table parent
  (`tests/test_retrieval.py:13-40`).

## Rui Ro Va Diem Can Chu Y

1. `table_parser.py` import `bs4` top-level (`table_parser.py:12`), nen bat ky
   import `services.chunking` nao cung fail neu chua cai `beautifulsoup4`.
2. `structure_parser.py` la deterministic heuristic. No hop voi van ban hanh
   chinh/quyet dinh mau, nhung layout nhieu cot tong quat, heading la, hoac OCR
   loi nang van la risk co chu dich.
3. Cross-page merge dua vao threshold 3.0 va cac tin hieu bbox/schema/parent. Neu
   bbox OCR thieu hoac page co title chen giua, table co the khong merge.
4. Course-transfer serializer nhan schema dua tren header co "mon" va marker
   "chuyen/da hoc/sv"; schema khac fallback an toan ve key-value, khong hallucinate.
5. Parent chunks co the dai hon max model token; validator co chu dich chi chan
   overflow o children vi parent khong duoc vector search.
6. `debug.py` mac dinh tai `SentenceTransformer`; offline debug phai explicit
   `--offline-tokenizer`.

## Test Command Da Thu

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
& 'C:\Users\MrEm\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s services/chunking/tests -v
```

Ket qua: 6 import errors, cung mot nguyen nhan `ModuleNotFoundError: No module
named 'bs4'`. Khong co test behavior nao duoc thuc thi trong runtime hien tai.

## Ket Luan Kien Truc

Phuong phap chunking trong thu muc nay la SAHC-v2:

```text
OCR JSON
-> normalize/filter/sort layout blocks
-> parse document tree
-> parse + reconstruct tables
-> create semantic atomic units
-> prefix-aware token packing
-> build parent and child chunks
-> validate
-> child-only embedding/search
-> parent/table/sibling context expansion
```

Gia tri chinh cua thiet ke la no chunk theo cau truc va y nghia truoc, roi moi
dung token budget de dong goi. Voi bang, moi row duoc serialize co schema va
context, thay vi embedding raw HTML. Voi retrieval, child nho giup hit chinh xac,
parent/table parent giup tra loi co du ngu canh.
