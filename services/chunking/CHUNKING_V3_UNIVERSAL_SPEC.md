# CHUNKING V3 — UNIVERSAL MULTI-DOCUMENT CHUNKING ARCHITECTURE

**Internal name:** `SAHC-v3`  
**Meaning:** **Structure-Aware Hierarchical Chunking v3 — Universal Multi-Document Architecture**  
**Primary implementation target:** `chunking/structure_parser.py`  
**Primary input:** OCR JSON with layout blocks  
**Core constraints:** deterministic, document-agnostic, no document classifier, no LLM API dependency.

---

# 0. Mục tiêu tài liệu

Tài liệu này nâng cấp đặc tả **SAHC-v2** thành **SAHC-v3**.

SAHC-v2 đã xác lập bốn năng lực cốt lõi phải được giữ nguyên:

1. **Structure-Aware Hierarchical Chunking**;
2. **Table-Aware Row Chunking**;
3. **Parent/Child Retrieval**;
4. **Token-Aware Packing**.

Thay đổi kiến trúc trung tâm ở v3 là:

> **Không xác định loại văn bản trước khi parse. Parser chỉ quan sát layout, heading grammar, numbering grammar, vị trí và quan hệ phân cấp để dựng cây tài liệu.**

Không được triển khai:

```python
doc_type = classify_document(...)

if doc_type == "CONTRACT":
    ...
elif doc_type == "DECISION":
    ...
```

Không được thêm:

```text
doc_classifier.py
```

Parser phải có thể xử lý bằng cùng một engine các tài liệu như:

```text
Quyết định
Tờ trình
Hợp đồng
Công văn
Phụ lục
Thông báo
Biên bản
Báo cáo
Quy chế
Quy định
```

mà không có nhánh logic theo `doc_type`.

---

# 1. Bối cảnh SAHC-v2 và lý do nâng cấp

SAHC-v2 đã thay pipeline cũ:

```text
OCR TXT
→ clean
→ split Điều bằng regex
→ embedding
→ Qdrant
```

bằng kiến trúc tốt hơn:

```text
OCR JSON
→ normalize layout blocks
→ reconstruct document structure
→ reconstruct cross-page tables
→ create semantic atomic units
→ token-aware packing
→ create parent/child chunks
→ build retrieval text
→ embed child chunks
→ Qdrant
```

Tuy nhiên `structure_parser.py` của v2 vẫn có trọng tâm lớn vào cấu trúc Quyết định:

```text
Điều
Khoản
Điểm
Căn cứ
Nơi nhận
```

Điều này chưa đủ tổng quát cho các dạng:

```text
Tờ trình:
    I. SỰ CẦN THIẾT
    1. Bối cảnh
    2. Cơ sở

Hợp đồng:
    ĐIỀU 3. THANH TOÁN
    3.1. Phương thức
    a) ...
    b) ...

Công văn:
    KÍNH GỬI:
    1. Nội dung...
    2. Đề nghị...

Phụ lục:
    PHỤ LỤC 01
    I. ...
    1. ...
    TABLE
```

SAHC-v3 giải quyết vấn đề này bằng một **Universal Hierarchical Grammar**.

---

# 2. Nguyên lý kiến trúc SAHC-v3

SAHC-v3 phải hoạt động dựa trên ba khái niệm tổng quát:

```text
ZONE
+
BOUNDARY CANDIDATE
+
HIERARCHICAL STACK
```

## 2.1 ZONE

Zone là vùng chức năng/layout của tài liệu:

```text
HEADER
BODY
CLOSING
ANNEX
```

Zone không phải document type.

## 2.2 BOUNDARY CANDIDATE

Một span OCR có thể là structural boundary nếu khớp một grammar tổng quát:

```text
PHẦN
CHƯƠNG
MỤC
I.
1.
Điều 1
Khoản 1
1.1.
Điểm a
a)
-
KÍNH GỬI:
BÊN A:
ĐIỀU KHOẢN CHUNG
...
```

## 2.3 HIERARCHICAL STACK

Parser duy trì một stack các node cấu trúc đang active:

```text
DOCUMENT
└── LEVEL 0
    └── LEVEL 1
        └── LEVEL 2
            └── LEVEL 3
                └── LEVEL 4
                    └── LEVEL 5
```

Boundary mới sẽ pop stack dựa trên **effective hierarchical level**, sau đó được attach vào ancestor phù hợp.

---

# 3. Design principles bắt buộc

## 3.1 Không dùng document classifier

Sai:

```python
if "HỢP ĐỒNG" in title:
    parse_contract(...)
```

Đúng:

```python
candidate = detect_boundary(span, layout_features)
level = resolve_effective_level(candidate, parser_state)
```

Ví dụ `HỢP ĐỒNG MUA BÁN` có thể được chọn làm document title vì:

```text
centered
uppercase
font lớn
block type title
position đầu tài liệu
```

không phải vì parser đã quyết định đây là Hợp đồng.

---

## 3.2 Deterministic và reproducible

Với cùng:

```text
OCR JSON
ChunkingConfig
parser_version
grammar_version
tokenizer_version
```

kết quả AST và chunks phải giống nhau.

Không dùng:

```text
LLM API
random thresholds
semantic boundary API
non-deterministic routing
```

---

## 3.3 Parse trước, pack sau

Không tạo chunk ngay trong lúc detect heading.

Luồng đúng:

```text
OCR blocks
→ logical spans
→ boundary detection
→ AST
→ atomic units
→ token-aware packing
→ parent/child chunks
```

---

## 3.4 Raw OCR bất biến

Duy trì song song:

```text
raw_text
normalized_text
match_text
```

Trong đó:

- `raw_text`: nguyên văn OCR, phục vụ audit/citation/debug;
- `normalized_text`: normalize nhẹ, dùng retrieval/parsing;
- `match_text`: representation chỉ dùng để match regex/grammar.

Không tự sửa nội dung pháp lý hay thương mại trong `raw_text`.

---

## 3.5 Explicit grammar ưu tiên hơn inferred layout

Ví dụ:

```text
ĐIỀU 3. THANH TOÁN
```

phải được nhận là `ARTICLE` trước khi xét generic `bold centered heading`.

Thứ tự tổng quát:

```text
explicit lexical boundary
>
numbering grammar
>
semantic/layout heading fallback
>
plain paragraph
```

---

## 3.6 Nominal level khác Effective level

Đây là nguyên lý quan trọng nhất để parser thật sự universal.

Ví dụ:

```text
I. SỰ CẦN THIẾT
1. Bối cảnh thực tế
```

thì:

```text
I.  → Level 1
1.  → Level 2
```

Nhưng:

```text
ĐIỀU 3. THANH TOÁN
1. Thanh toán đợt 1
2. Thanh toán đợt 2
```

nếu luôn coi `1.` là Level 2 thì sẽ phá hierarchy của `Điều 3`.

Do đó mỗi candidate cần:

```python
candidate.nominal_level
candidate.effective_level
```

Registry gán `nominal_level`.

State machine resolve `effective_level` từ:

```text
active structural stack
numbering depth
numbering sequence
indent/layout
explicit parent đang active
```

không cần document type.

---

# 4. Kiến trúc tổng thể

## 4.1 End-to-end flow

```text
documents.csv / document metadata
             │
             ▼
        OCR JSON
             │
             ▼
      ocr_parser.py
             │
             ▼
       OCRBlock[]
             │
             ▼
       normalize.py
             │
             ├── content_raw
             ├── content_normalized
             └── match_text
             │
             ▼
  reading-order normalization
             │
             ▼
  span/paragraph segmentation
             │
             ▼
      layout feature extraction
             │
             ▼
     zone signal detection
   HEADER / BODY / CLOSING
             │
             ▼
  boundary candidate registry
             │
             ▼
 candidate scoring + precedence
             │
             ▼
 effective-level resolution
             │
             ▼
   deterministic stack machine
             │
             ▼
       DocumentNode AST
             │
      ┌──────┴─────────┐
      │                │
      ▼                ▼
 text leaves      table fragments
      │                │
      │                ▼
      │          table_parser.py
      │                │
      │         cross-page merge
      │                │
      │         flattened headers
      │                │
      │        semantic table rows
      │                │
      └──────┬─────────┘
             ▼
        AtomicUnit[]
             │
             ▼
      token_packer.py
             │
             ▼
 parent/child chunk builder
             │
             ▼
 retrieval_text builder
             │
             ▼
 tokenizer validation
             │
             ▼
 embedding child chunks
             │
             ▼
        Qdrant V3
             │
             ▼
      child retrieval
             │
             ▼
 parent/sibling expansion
             │
             ▼
           RAG LLM
```

---

## 4.2 AST ví dụ — Hợp đồng

```text
DOCUMENT: "HỢP ĐỒNG MUA BÁN"
│
├── METADATA
│   ├── Số/Ký hiệu
│   ├── Date
│   └── Header fields
│
├── SECTION L1: "BÊN A"
│   └── paragraphs
│
├── SECTION L1: "BÊN B"
│   └── paragraphs
│
├── ARTICLE L2: "Điều 1. Đối tượng hợp đồng"
│   ├── PARAGRAPH
│   ├── CLAUSE L3: "1.1. ..."
│   └── CLAUSE L3: "1.2. ..."
│
├── ARTICLE L2: "Điều 3. Thanh toán"
│   ├── CLAUSE L3: "3.1. Phương thức"
│   │   ├── POINT L4: "a) ..."
│   │   └── POINT L4: "b) ..."
│   └── TABLE
│       ├── TABLE_ROW
│       └── TABLE_ROW
│
├── CLOSING
│   ├── SIGNATURE
│   └── RECIPIENTS
│
└── ANNEX L0: "PHỤ LỤC 01"
    ├── SECTION L1: "I. ..."
    └── TABLE
```

---

## 4.3 AST ví dụ — Tờ trình không có Điều

```text
DOCUMENT: "TỜ TRÌNH"
│
├── METADATA
├── SECTION L1: "KÍNH GỬI: ..."
├── SECTION L1: "I. SỰ CẦN THIẾT"
│   ├── SECTION L2: "1. Bối cảnh thực tế"
│   └── SECTION L2: "2. Căn cứ"
├── SECTION L1: "II. NỘI DUNG ĐỀ XUẤT"
│   ├── SECTION L2: "1. Phương án"
│   └── SECTION L2: "2. Kinh phí"
└── CLOSING
```

---

## 4.4 AST ví dụ — Quyết định

```text
DOCUMENT: "QUYẾT ĐỊNH"
│
├── PREAMBLE
│   ├── LEGAL_BASIS
│   ├── LEGAL_BASIS
│   └── LEGAL_BASIS
├── ARTICLE L2: "Điều 1"
│   ├── PARAGRAPH
│   └── TABLE
├── ARTICLE L2: "Điều 2"
│   └── CLAUSE L3: "Khoản 1"
└── CLOSING
    ├── RECIPIENTS
    └── SIGNATURE
```

---

# 5. Package structure đề xuất

Giữ kiến trúc modular của v2 và bổ sung các thành phần generic:

```text
chunking/
├── __init__.py
├── config.py
├── models.py
├── normalize.py
├── ocr_parser.py
├── reading_order.py
├── span_segmenter.py
├── layout_features.py
├── boundary_registry.py
├── boundary_scoring.py
├── structure_parser.py
├── table_parser.py
├── table_serializers.py
├── token_counter.py
├── token_packer.py
├── chunk_builder.py
├── retrieval_text.py
├── parent_store.py
├── validators.py
└── debug.py
```

Tests:

```text
tests/
├── test_normalize_match_text.py
├── test_span_segmenter.py
├── test_boundary_registry.py
├── test_boundary_precedence.py
├── test_structure_parser_generic.py
├── test_structure_parser_contract.py
├── test_structure_parser_submission.py
├── test_structure_parser_decision.py
├── test_structure_parser_official_letter.py
├── test_structure_parser_annex.py
├── test_false_split_avoidance.py
├── test_table_parser.py
├── test_cross_page_table.py
├── test_token_packer.py
├── test_chunk_builder.py
└── test_end_to_end_chunking_v3.py
```

Không đặt toàn bộ v3 vào một file `embedding.py`.

---

# 6. Core data models

## 6.1 OCRBlock

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class OCRBlock:
    page_number: int
    block_index: int
    block_type: str

    bbox: tuple[float, float, float, float] | None

    content_raw: str
    content_normalized: str

    angle: float | int | None = None

    # Optional nếu OCR cung cấp.
    font_size: float | None = None
    is_bold: bool | None = None
    is_italic: bool | None = None
    text_align: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)
```

Không bắt buộc style fields phải tồn tại.

Nếu OCR hiện tại chỉ có:

```text
page_number
type
bbox
content
angle
```

parser vẫn phải hoạt động.

---

## 6.2 LogicalSpan

Một OCR block có thể chứa nhiều boundary logic.

Ví dụ:

```text
NỘI DUNG:
1. Phương án A
2. Phương án B
```

Nếu chỉ match đầu block thì parser sẽ bỏ sót `1.` và `2.`.

Do đó thêm intermediate model:

```python
@dataclass
class LogicalSpan:
    span_id: str

    page_number: int
    source_block_id: str
    source_block_type: str

    raw_text: str
    normalized_text: str
    match_text: str

    bbox: tuple[float, float, float, float] | None

    block_start: bool
    paragraph_start: bool
    line_start: bool

    char_start: int
    char_end: int

    layout: dict
```

Phải giữ `char_start/char_end` để audit về block gốc.

---

## 6.3 BoundaryCandidate

```python
@dataclass
class BoundaryCandidate:
    kind: str
    label: str | None

    nominal_level: int | None
    effective_level: int | None

    title_text: str
    numbering_key: str | None

    base_priority: int
    confidence: float

    evidence: dict
    source_span_id: str
```

Ví dụ `kind`:

```text
annex
part
chapter
section_word
roman
semantic_heading
style_heading
article
clause_word
decimal_number
primary_number
point_word
letter_point
bullet
recipients
signature
legal_basis
metadata_label
```

---

## 6.4 DocumentNode

```python
@dataclass
class DocumentNode:
    node_id: str

    node_type: str
    boundary_kind: str | None

    title: str | None
    level: int | None

    text_raw: str
    text_normalized: str

    page_start: int
    page_end: int

    parent_id: str | None
    children_ids: list[str]

    section_path: list[str]

    source_block_ids: list[str]
    metadata: dict
```

`node_type` tối thiểu:

```text
document
metadata
header
preamble
legal_basis
section
article
clause
point
bullet
paragraph
table
table_row
recipients
signature
annex
other
```

`node_type` mô tả cấu trúc, không mô tả document type.

---

## 6.5 AtomicUnit

Giữ từ v2:

```python
@dataclass
class AtomicUnit:
    unit_id: str
    unit_type: str

    parent_id: str
    section_path: list[str]

    raw_text: str
    normalized_text: str

    page_start: int
    page_end: int

    source_block_ids: list[str]
    metadata: dict
```

Ví dụ:

```text
legal_basis
section_intro
article_intro
clause
point
paragraph
table_row
recipient_item
signature
```

---

## 6.6 Chunk

```python
@dataclass
class Chunk:
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
    metadata: dict
```

Metadata v3 nên có:

```text
node_id
node_type
node_level
boundary_kind
boundary_confidence
table_id
table_row_index
sibling_group_id
parser_version
grammar_version
```

---

# 7. Normalization cho deterministic grammar matching

## 7.1 Ba representation văn bản

Mỗi span phải có:

```text
raw_text
normalized_text
match_text
```

### raw_text

Giữ nguyên OCR.

### normalized_text

Cho phép:

```text
Unicode normalization
NBSP → space
collapse repeated horizontal whitespace
collapse excessive blank lines
safe trim
```

Không cho phép:

```text
guess spelling correction
invent missing legal values
rewrite names/numbers
```

### match_text

Chỉ dùng cho parser.

Ví dụ:

```python
import re
import unicodedata

def make_match_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("Đ", "D").replace("đ", "d")

    decomposed = unicodedata.normalize("NFD", text)
    text = "".join(
        ch for ch in decomposed
        if unicodedata.category(ch) != "Mn"
    )

    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

Nhờ đó parser nhận được:

```text
Điều
ĐIỀU
Dieu
dieu
```

mà không sửa text được lưu trong chunk.

---

## 7.2 OCR spacing noise

OCR có thể tạo:

```text
Đ I Ề U  3
K H O Ả N  1
N Ơ I  N H Ậ N
```

Không được globally remove toàn bộ spaces.

Chỉ boundary-specific matcher được phép compact một prefix ngắn:

```python
compact_prefix = re.sub(r"\s+", "", prefix)
```

Ví dụ parser có thể so:

```text
d i e u 3
```

với label `dieu`.

Raw OCR vẫn giữ nguyên.

---

# 8. Reading order và layout features

## 8.1 Reading order baseline

Trên từng page:

```text
bbox.y1
then bbox.x1
```

Nhưng cần special handling tối thiểu cho header hai cột.

Khuyến nghị:

```text
1. Cluster blocks theo horizontal band.
2. Với top-header band, cho phép left/right column grouping.
3. Sau header, dùng y-major reading order.
4. Table block giữ nguyên atomic layout block.
```

---

## 8.2 Normalized bbox

Parser nên sử dụng:

```text
x1, y1, x2, y2 ∈ [0, 1]
```

Nếu bbox là absolute và biết page size:

```python
x1 /= page_width
x2 /= page_width
y1 /= page_height
y2 /= page_height
```

Nếu không biết page size, không được đoán.

Log warning và tắt bbox-dependent bonuses.

---

## 8.3 Layout features

Mỗi span derive:

```python
layout = {
    "x1": ...,
    "y1": ...,
    "x2": ...,
    "y2": ...,
    "width": ...,
    "height": ...,
    "center_x": ...,

    "is_top_20pct": ...,
    "is_bottom_25pct": ...,

    "is_centered": ...,
    "is_left_aligned": ...,

    "uppercase_ratio": ...,
    "line_length": ...,
    "word_count": ...,

    "is_title_block": ...,
    "is_bold": ...,
    "font_size": ...,
    "font_size_ratio": ...,
}
```

Nếu style metadata không có, chỉ dùng các feature có thể xác định chắc chắn.

`is_centered` có thể ước lượng:

```python
abs(center_x - 0.5) <= centered_tolerance
```

nhưng chỉ là supporting evidence.

---

# 9. Universal document zones

## 9.1 HEADER

Primary evidence:

```text
page == 1
and bbox.y1 < 0.20
```

Có thể gồm:

```text
Quốc hiệu
Tiêu ngữ
Tên cơ quan
Số/Ký hiệu
Ngày
Địa điểm
Tên văn bản
Thông tin các bên
```

Nhưng:

> Không được tự động biến toàn bộ top 20% thành metadata.

Một title nổi bật vẫn phải có thể trở thành document title.

---

## 9.2 BODY

BODY là zone mặc định cho:

```text
title
preamble
section
article
clause
point
paragraph
table
party information
proposal content
official-letter content
```

---

## 9.3 CLOSING

Closing được kích hoạt bởi strong structural signal, không phải chỉ do nằm cuối page.

Signals:

```text
Nơi nhận:
KT.
TL.
TUQ.
TM.
ĐẠI DIỆN
NGƯỜI KÝ
GIÁM ĐỐC
TỔNG GIÁM ĐỐC
CHỦ TỊCH
signature-like figure/caption
```

Location bonus:

```text
last page
and bbox.y1 > 0.60
```

Nhưng location một mình không đủ.

---

## 9.4 ANNEX

`PHỤ LỤC` là structural boundary, không phải closing noise.

Ví dụ:

```text
SIGNATURE
...
PHỤ LỤC 01
BẢNG DANH MỤC...
```

Parser phải cho phép transition:

```text
CLOSING
→ ANNEX
```

và mở subtree mới.

---

# 10. Universal Hierarchical Boundary Registry

Canonical levels:

```text
Level 0 — PHẦN / CHƯƠNG / PHỤ LỤC root-like boundary
Level 1 — MỤC / Roman numeral / prominent semantic heading
Level 2 — Điều / primary ordinal
Level 3 — Khoản / decimal numbering
Level 4 — Điểm / alphabetic numbering
Level 5 — bullet
```

Đây là **nominal levels**.

State machine có quyền resolve thành effective levels khi numbering ambiguous.

---

# 11. Universal Boundary Priority Matrix

Parser match chủ yếu trên `match_text`.

| Priority | Boundary kind | Nominal level | Ví dụ | Regex/pattern concept | Guard chính |
|---:|---|---:|---|---|---|
| 10 | `annex` | 0 | `PHỤ LỤC`, `PHỤ LỤC 01` | `^phu\s*luc\b...` | paragraph start, short heading/layout |
| 20 | `part` | 0 | `PHẦN I`, `PHẦN THỨ HAI` | `^phan\s+(?:thu\s+)?...` | span start |
| 30 | `chapter` | 0 | `CHƯƠNG I`, `CHƯƠNG 2` | `^chuong\s+...` | span start |
| 40 | `section_word` | 1 | `MỤC 1`, `MỤC II` | `^muc\s+...` | span start |
| 50 | `article` | 2 | `Điều 3`, `ĐIỀU 3. THANH TOÁN` | `^dieu\s+\d+[a-z]?...` | not inline citation |
| 60 | `clause_word` | 3 | `Khoản 1`, `Khoản 2.1` | `^khoan\s+...` | span start |
| 70 | `point_word` | 4 | `Điểm a`, `Điểm b` | `^diem\s+...` | span start |
| 80 | `decimal_number` | 3 default | `1.1.`, `3.2.1.` | `^\d+(?:\.\d+)+...` | compatible parent/context |
| 90 | `roman` | 1 | `I. SỰ CẦN THIẾT` | `^[ivxlcdm]+\s*[.)]...` | heading-like |
| 100 | `primary_number` | 2 default | `1. Bối cảnh` | `^\d{1,3}\s*[.)]...` | high ambiguity threshold |
| 110 | `letter_point` | 4 | `a)`, `b.`, `đ)` | `^[a-zđ]\s*[.)]...` | active parent/context |
| 120 | `semantic_heading` | 1 default | `KÍNH GỬI:`, `BÊN A:`, `ĐIỀU KHOẢN CHUNG` | lexical heading registry | heading-shape evidence |
| 130 | `style_heading` | 1 default | bold/centered uppercase | layout score | high confidence only |
| 140 | `bullet` | 5 | `-`, `+`, `*` | `^[-+*•▪–—]\s+` | span start |
| 150 | `plain_paragraph` | — | prose | fallback | no accepted boundary |

Priority số nhỏ hơn được evaluate trước.

---

# 12. Regex/matcher specification

Regex dưới đây là reference implementation, có thể tinh chỉnh nhưng phải giữ nguyên behavior và test coverage.

## 12.1 Annex

```python
ANNEX_RE = re.compile(
    r"^\s*phu\s*luc"
    r"(?:\s+(?:so\s*)?[0-9ivxlcdm]+)?"
    r"\s*[:.\-]?\s*",
    re.IGNORECASE,
)
```

---

## 12.2 Part

```python
PART_RE = re.compile(
    r"^\s*phan\s+"
    r"(?:(?:thu)\s+)?"
    r"([0-9]+|[ivxlcdm]+|mot|hai|ba|tu|nam|sau|bay|tam|chin|muoi)"
    r"\s*[:.\-]?\s*",
    re.IGNORECASE,
)
```

Không cần NLP ordinal phức tạp; các biến thể phải explicit và deterministic.

---

## 12.3 Chapter

```python
CHAPTER_RE = re.compile(
    r"^\s*chuong\s+([0-9]+|[ivxlcdm]+)"
    r"\s*[:.\-]?\s*",
    re.IGNORECASE,
)
```

---

## 12.4 Mục

```python
SECTION_RE = re.compile(
    r"^\s*muc\s+([0-9]+|[ivxlcdm]+)"
    r"\s*[:.\-]?\s*",
    re.IGNORECASE,
)
```

---

## 12.5 Điều

```python
ARTICLE_RE = re.compile(
    r"^\s*dieu\s+(\d+[a-z]?)"
    r"\s*[:.\-]?\s*",
    re.IGNORECASE,
)
```

Phải nhận:

```text
Điều 1:
Điều 1.
Điều 1
ĐIỀU 1
Dieu 1
```

Không match inline mention.

---

## 12.6 Khoản

```python
CLAUSE_WORD_RE = re.compile(
    r"^\s*khoan\s+(\d+(?:\.\d+)*)"
    r"\s*[:.\-]?\s*",
    re.IGNORECASE,
)
```

---

## 12.7 Điểm

```python
POINT_WORD_RE = re.compile(
    r"^\s*diem\s+([a-z]|d|\d+)"
    r"\s*[:.\-]?\s*",
    re.IGNORECASE,
)
```

Vì `match_text` map `đ → d`, display label phải lấy từ `normalized_text` gốc.

---

## 12.8 Decimal numbering

```python
DECIMAL_RE = re.compile(
    r"^\s*(\d+(?:\.\d+){1,4})"
    r"\s*[.)\-:]?\s+"
    r"(?=\S)",
)
```

Ví dụ:

```text
1.1. Nội dung
3.2.1 Trách nhiệm
2.4.5.1 Yêu cầu
```

Không hỗ trợ depth vô hạn.

---

## 12.9 Roman heading

```python
ROMAN_RE = re.compile(
    r"^\s*([ivxlcdm]{1,8})"
    r"\s*[.)]\s+"
    r"(?=\S)",
    re.IGNORECASE,
)
```

Bắt buộc có delimiter.

---

## 12.10 Primary ordinal

```python
PRIMARY_NUMBER_RE = re.compile(
    r"^\s*(\d{1,3})"
    r"\s*[.)]\s+"
    r"(?=\S)",
)
```

Đây là pattern ambiguous, bắt buộc qua scoring và effective-level resolver.

---

## 12.11 Letter point

```python
LETTER_POINT_RE = re.compile(
    r"^\s*([a-z])"
    r"\s*[.)]\s+"
    r"(?=\S)",
    re.IGNORECASE,
)
```

Với `đ)`, dùng parser-prefix normalization trước khi match.

---

## 12.12 Bullet

```python
BULLET_RE = re.compile(
    r"^\s*[-+*•▪–—]\s+(?=\S)"
)
```

---

# 13. Semantic Heading Registry

Một universal parser có thể dùng registry các heading phổ biến trong văn bản chính thức.

Đây **không phải document classification**.

Nó chỉ tương đương với việc nhận biết lexical marker như `Điều`, `Mục`, `Nơi nhận`.

Ví dụ config:

```python
SEMANTIC_HEADING_PREFIXES = {
    "kinh gui",
    "noi dung",
    "noi dung de nghi",
    "noi dung to trinh",
    "su can thiet",
    "co so phap ly",
    "co so thuc tien",
    "muc dich",
    "pham vi",
    "doi tuong",
    "to chuc thuc hien",
    "dieu khoan chung",
    "dieu khoan thi hanh",
    "trach nhiem cac ben",
    "quyen va nghia vu",
    "ben a",
    "ben b",
    "ben c",
    "dai dien ben a",
    "dai dien ben b",
}
```

Rules:

1. Registry chỉ chứa structural labels.
2. Match registry không kích hoạt parser riêng.
3. Match vẫn cần heading-shape evidence.
4. Registry phải nằm trong config, không hard-code rải rác.
5. Mỗi label mới cần test.

Ví dụ:

```text
BÊN A: CÔNG TY ABC
```

có thể mở Level-1 section.

Parser không tạo:

```text
document_type = contract
```

---

# 14. Style-based Heading Fallback

Nếu không match lexical/numbering boundary, một span có thể trở thành `style_heading`.

Đây là fallback cuối cùng trước `plain_paragraph`.

Score gợi ý:

```text
+0.25 block_type == "title"
+0.20 is_bold is True
+0.15 font_size_ratio >= 1.20
+0.15 is_centered
+0.10 uppercase_ratio >= 0.75
+0.10 word_count <= 14
+0.05 ends_with(":")
```

Negative:

```text
-0.25 word_count > 30
-0.30 sentence-like punctuation
-0.30 text length > 220 chars
```

Threshold mặc định:

```python
STYLE_HEADING_THRESHOLD = 0.75
```

Nếu font/bold không có trong OCR JSON, phải yêu cầu evidence còn lại mạnh hơn.

Không được coi một paragraph dài chỉ vì centered là heading.

---

# 15. Boundary Scoring

## 15.1 Tại sao cần scoring

Regex:

```text
1.
```

không đủ để xác định boundary.

Nó có thể là:

```text
list item
clause
section
OCR artifact
sentence continuation
```

Do đó scoring phải kết hợp:

```text
regex
paragraph position
layout
span shape
active hierarchy
neighbor spans
zone
numbering sequence
```

---

## 15.2 Score reference

Các số dưới đây là config defaults, không phải magic numbers bất biến.

```python
score = pattern_base_score

if span.paragraph_start:
    score += 0.15

if span.block_start:
    score += 0.05

if span.source_block_type == "title":
    score += 0.15

if layout.is_centered:
    score += 0.08

if layout.is_bold:
    score += 0.08

if layout.uppercase_ratio >= 0.75:
    score += 0.05

if layout.word_count <= 14:
    score += 0.05
```

Negative evidence:

```python
if looks_like_mid_sentence(span):
    score -= 0.60

if looks_like_inline_legal_reference(span):
    score -= 0.55

if layout.word_count > 45:
    score -= 0.15

if span.normalized_text.startswith((",", ";")):
    score -= 0.20
```

Clamp:

```python
score = max(0.0, min(1.0, score))
```

---

## 15.3 Threshold defaults

```python
EXPLICIT_KEYWORD_THRESHOLD = 0.50
DECIMAL_THRESHOLD = 0.65
ROMAN_THRESHOLD = 0.70
PRIMARY_NUMBER_THRESHOLD = 0.72
LETTER_POINT_THRESHOLD = 0.68
STYLE_HEADING_THRESHOLD = 0.75
```

Explicit labels như:

```text
Điều
Khoản
Mục
Chương
Phần
Phụ lục
```

có threshold thấp hơn vì ít ambiguous hơn.

---

# 16. Universal Precedence

Nếu một span match nhiều candidate, chọn theo precedence:

```text
ANNEX
>
PART
>
CHAPTER
>
SECTION_WORD
>
ARTICLE
>
CLAUSE_WORD
>
POINT_WORD
>
DECIMAL_NUMBER
>
ROMAN
>
PRIMARY_NUMBER
>
LETTER_POINT
>
SEMANTIC_HEADING
>
STYLE_HEADING
>
BULLET
>
PARAGRAPH
```

Ví dụ:

```text
Điều 3. Thanh toán
```

phải là `ARTICLE`, không phải generic primary number.

```text
Khoản 1. ...
```

phải là `CLAUSE_WORD`.

```text
PHỤ LỤC 01
```

phải là `ANNEX`, không phải style heading.

---

# 17. False-Split Avoidance

Đây là phần bắt buộc của SAHC-v3.

## 17.1 Chỉ detect ở structural start

Valid:

```text
Điều 2. Trách nhiệm...
```

Invalid:

```text
Theo Điều 2 của Luật...
```

vì `Điều` không nằm ở start của logical span.

---

## 17.2 Inline legal-reference guard

Reject hoặc giảm score mạnh với các prefix:

```text
theo Điều 2
tại Điều 3
quy định tại Khoản 1
theo Khoản 2
nêu tại Điểm a
căn cứ Điều 4
```

Parser-only pattern:

```python
INLINE_REFERENCE_PREFIX_RE = re.compile(
    r"^\s*(theo|tai|can cu|quy dinh tai|duoc quy dinh tai|neu tai)\s+"
    r"(dieu|khoan|diem)\b",
    re.IGNORECASE,
)
```

Ví dụ:

```text
Theo Điều 2 Luật Doanh nghiệp, ...
```

là paragraph, không phải Article.

---

## 17.3 Sentence-continuation guard

OCR có thể xuống dòng:

```text
Bên A thanh toán theo
Điều 2 của hợp đồng này.
```

Dòng thứ hai bắt đầu bằng `Điều 2` nhưng là continuation.

Dùng evidence:

```text
previous span chưa kết thúc câu
vertical gap nhỏ
cùng x-indent
cùng source block
same font/style
current span.paragraph_start == False
```

Sau đó không mở Article.

---

## 17.4 Long-line guard

Không reject boundary chỉ vì span dài.

Ví dụ:

```text
1. Trong trường hợp Bên A không thanh toán đúng hạn, Bên B có quyền ...
```

vẫn có thể là clause.

Long text chỉ giảm confidence cho generic numbering, không vô hiệu explicit boundary.

---

## 17.5 Table-context guard

Không chạy document numbering grammar lên HTML cell trước khi parse table.

Cell:

```text
1.
```

không được mở section.

`block_type == "table"` phải route sang table parser.

---

## 17.6 Page-number guard

Standalone text như:

```text
1
2
10
```

ở sát bottom/top page, font nhỏ, không có title evidence phải được xem như page number/noise, không phải structural heading.

---

## 17.7 Numeric-value guard

Không coi các giá trị như:

```text
1.000.000 đồng
2.5%
3.14
01.08.2026
```

là hierarchy.

Generic numbering regex phải yêu cầu heading/list delimiter và textual content phù hợp.

---

# 18. Effective-Level Resolution

## 18.1 Interface

```python
def resolve_effective_level(
    candidate: BoundaryCandidate,
    stack: list[DocumentNode],
    previous_boundary: BoundaryCandidate | None,
) -> int | None:
    ...
```

Không nhận `doc_type`.

---

## 18.2 Explicit levels giữ ổn định

Các kind:

```text
annex
part
chapter
section_word
article
clause_word
point_word
```

thường giữ nominal level.

```python
if candidate.kind in EXPLICIT_LEVEL_KINDS:
    return candidate.nominal_level
```

---

## 18.3 Primary number demotion

Nominal:

```text
1. → Level 2
```

Nếu active stack có explicit Article Level 2:

```text
Điều 3
1. ...
2. ...
```

thì generic `1.`/`2.` phải trở thành Level 3.

```python
if candidate.kind == "primary_number":
    active_article = nearest_active_kind(stack, "article")
    if active_article is not None:
        return min(active_article.level + 1, 5)
```

Nhưng:

```text
I. SỰ CẦN THIẾT
1. Bối cảnh
```

không có Article, nên `1.` giữ Level 2.

---

## 18.4 Decimal numbering

Default:

```text
3.1     → L3
3.1.1   → L4
3.1.1.1 → L5
```

Reference:

```python
parts = candidate.numbering_key.split(".")
depth = len(parts)

if depth == 2:
    level = 3
elif depth == 3:
    level = 4
else:
    level = min(5, depth + 1)
```

Nếu active Article number = `3` và candidate = `3.1`, confidence tăng.

Nếu không match, không reject ngay vì một số văn bản restart numbering.

---

## 18.5 Letter point resolution

Nominal:

```text
a) → L4
```

Nếu không có parent Level 0..3 hợp lệ:

```text
1. thử contextual re-leveling;
2. nếu không chắc, downgrade thành list item/paragraph;
3. log orphan_boundary_downgraded.
```

Không tạo:

```text
DOCUMENT
└── Level 4
```

một cách mù quáng.

---

## 18.6 Bullet resolution

Bullet nominal Level 5.

Nếu active parent là Level 3:

```text
attach Level 4/5 list-like node tùy policy
```

Nếu active parent chỉ là document root:

```text
keep as paragraph/list item
```

Không ép hierarchy sâu giả tạo.

---

## 18.7 Orphan prevention invariant

Không được tạo structural orphan vô lý.

Nếu candidate level quá sâu mà không có ancestor phù hợp:

```text
resolve upward/downward theo active context
hoặc downgrade thành paragraph/list item
```

Log:

```text
orphan_boundary_downgraded
```

---

# 19. Numbering Sequence Heuristics

Sequence là deterministic evidence:

```text
I. → II. → III.
1. → 2. → 3.
3.1 → 3.2 → 3.3
a) → b) → c)
```

Helper:

```python
def numbering_sequence_score(
    previous: BoundaryCandidate | None,
    current: BoundaryCandidate,
) -> float:
    ...
```

Valid sequence tăng confidence.

Ví dụ:

```text
1. → 2.
```

có thể +0.10.

Jump:

```text
1. → 9.
```

không reject, chỉ không có sequence bonus.

Sequence không được override false-split guard.

---

# 20. Deterministic State Machine cho `structure_parser.py`

## 20.1 ParserState

```python
@dataclass
class ParserState:
    document_root_id: str

    current_zone: str

    stack: list[str]
    current_structural_node_id: str

    previous_span_id: str | None
    previous_boundary: BoundaryCandidate | None

    closing_started: bool
    annex_started: bool
```

---

## 20.2 Stack invariant

Stack chỉ chứa structural nodes có level tăng dần.

Ví dụ:

```text
[
    DOCUMENT,
    L0 CHAPTER,
    L1 MỤC,
    L2 ARTICLE,
    L3 CLAUSE,
]
```

Paragraph/table row không nằm trên structural stack.

---

## 20.3 Main algorithm

```python
def build_document_ast(
    blocks: list[OCRBlock],
    config: ChunkingConfig,
) -> DocumentNode:
    ordered = sort_blocks_in_reading_order(blocks)

    spans = []
    for block in ordered:
        if block.block_type == "table":
            spans.append(make_table_placeholder(block))
        else:
            spans.extend(segment_block_into_logical_spans(block))

    features = compute_layout_features(spans)

    root = create_document_root(
        title=detect_document_title(spans, features, config),
    )

    state = ParserState(
        document_root_id=root.node_id,
        current_zone="HEADER",
        stack=[root.node_id],
        current_structural_node_id=root.node_id,
        previous_span_id=None,
        previous_boundary=None,
        closing_started=False,
        annex_started=False,
    )

    for span in spans:
        if span.is_table_placeholder:
            attach_table_fragment(
                span=span,
                parent_id=state.current_structural_node_id,
            )
            continue

        zone_signal = detect_zone_signal(
            span=span,
            state=state,
            features=features,
        )

        update_zone_state(
            zone_signal=zone_signal,
            state=state,
        )

        candidate = detect_best_boundary_candidate(
            span=span,
            state=state,
            features=features,
            config=config,
        )

        if candidate is None:
            attach_paragraph(
                span=span,
                parent_id=state.current_structural_node_id,
                zone=state.current_zone,
            )
            state.previous_span_id = span.span_id
            continue

        if candidate.kind == "annex":
            close_open_body_nodes(state)

            annex_node = open_annex_node(
                candidate=candidate,
                root=root,
                span=span,
            )

            state.stack = [root.node_id, annex_node.node_id]
            state.current_structural_node_id = annex_node.node_id
            state.current_zone = "ANNEX"
            state.annex_started = True
            state.previous_boundary = candidate
            continue

        if candidate.kind == "recipients":
            close_open_body_nodes(state)
            node = open_or_get_recipients_node(span, root, state)
            state.current_structural_node_id = node.node_id
            state.current_zone = "CLOSING"
            state.closing_started = True
            continue

        if candidate.kind == "signature":
            close_open_body_nodes(state)
            node = open_or_get_signature_node(span, root, state)
            state.current_structural_node_id = node.node_id
            state.current_zone = "CLOSING"
            state.closing_started = True
            continue

        if candidate.kind == "legal_basis":
            attach_legal_basis(span, root, state)
            continue

        if is_structural_boundary(candidate):
            level = resolve_effective_level(
                candidate=candidate,
                stack=get_nodes(state.stack),
                previous_boundary=state.previous_boundary,
            )

            if level is None:
                attach_paragraph(
                    span=span,
                    parent_id=state.current_structural_node_id,
                    zone=state.current_zone,
                )
                continue

            candidate.effective_level = level

            while top_structural_level(state.stack) >= level:
                pop_structural_node(state.stack)

            parent_id = nearest_structural_parent(
                stack=state.stack,
                child_level=level,
            )

            node = create_structural_node(
                candidate=candidate,
                parent_id=parent_id,
                span=span,
            )

            append_child(parent_id, node.node_id)

            state.stack.append(node.node_id)
            state.current_structural_node_id = node.node_id
            state.current_zone = (
                "ANNEX" if state.annex_started else "BODY"
            )
            state.previous_boundary = candidate
            continue

        attach_paragraph(
            span=span,
            parent_id=state.current_structural_node_id,
            zone=state.current_zone,
        )

    finalize_page_ranges(root)
    build_section_paths(root)
    return root
```

---

# 21. Zone State Behavior

## 21.1 HEADER → BODY

Transition khi gặp một trong các signal đủ mạnh:

```text
prominent document title
first accepted body structural heading
first long body paragraph dưới header zone
legal-basis sequence
```

Không giữ parser ở HEADER chỉ vì layout đầu trang bất thường.

---

## 21.2 BODY → CLOSING

Transition khi accepted:

```text
Nơi nhận
signature heading
signature figure + caption evidence
```

Không dựa duy nhất vào `last_page + y > 0.60`.

---

## 21.3 CLOSING → ANNEX

Khi accepted `PHỤ LỤC`:

```text
reset structural stack
DOCUMENT
└── ANNEX
```

Nội dung sau đó được parse bình thường bằng cùng grammar.

---

# 22. Document Title Detection

`section_path` cần root title ổn định.

Dùng deterministic scorer trên page 1.

Positive evidence:

```text
block_type == title
centered
bold
font lớn
uppercase
short/medium text
position phù hợp
```

Negative evidence:

```text
Quốc hiệu
Tiêu ngữ
Số:
date/place line
agency header
standalone page number
```

Ví dụ title có thể là:

```text
QUYẾT ĐỊNH
TỜ TRÌNH
HỢP ĐỒNG MUA BÁN
CÔNG VĂN VỀ VIỆC ...
PHỤ LỤC 01
```

Nếu không detect chắc chắn:

```text
fallback document metadata Summary
```

Nếu Summary cũng rỗng:

```text
không bịa root title
```

---

# 23. Header Metadata Handling

Header metadata không nằm trên structural stack.

AST:

```text
DOCUMENT
├── METADATA
└── BODY...
```

Possible metadata:

```text
Số:
Số/Ký hiệu:
Ký hiệu:
Ngày:
Tên cơ quan
Quốc hiệu/Tiêu ngữ
```

Không embedding từng line header thành child riêng mặc định.

Chỉ đưa metadata có ích vào retrieval prefix.

---

# 24. Legal-Basis Handling

Giữ capability v2.

Parser-only prefixes:

```text
Căn cứ
Căn cử
Căn cứ vào
Theo đề nghị
Xét đề nghị
```

Config:

```python
LEGAL_BASIS_PREFIXES = (
    "can cu",
    "can cu vao",
    "can cu tren co so",
    "theo de nghi",
    "xet de nghi",
)
```

Chỉ tạo `legal_basis` khi span có behavior của preamble item.

Không coi mọi câu chứa `căn cứ` là legal basis.

Ví dụ:

```text
Phương án được xây dựng căn cứ vào kết quả khảo sát...
```

ở giữa một section vẫn là paragraph.

---

# 25. Generic `section_path`

## 25.1 Nguyên tắc

Không hard-code:

```python
if contract:
    path = ...
```

Path được derive từ ancestry của AST.

---

## 25.2 Algorithm

```python
PATH_NODE_TYPES = {
    "document",
    "annex",
    "section",
    "article",
    "clause",
    "point",
    "table",
}


def build_section_path(
    node: DocumentNode,
    node_index: dict[str, DocumentNode],
) -> list[str]:
    lineage = get_ancestors_including_self(node, node_index)

    path = []

    for item in lineage:
        if item.node_type not in PATH_NODE_TYPES:
            continue

        label = canonical_path_label(item)

        if not label:
            continue

        if path and normalize_key(path[-1]) == normalize_key(label):
            continue

        path.append(label)

    return path
```

---

## 25.3 Canonical path label

Ưu tiên giữ heading thật:

```text
Điều 3. Thanh toán
I. SỰ CẦN THIẾT
1. Bối cảnh thực tế
Khoản 3.1
```

Nếu heading quá dài, chỉ được compact theo rule cơ học, không semantic rewrite.

---

## 25.4 Required examples

### Hợp đồng

```json
[
  "HỢP ĐỒNG MUA BÁN",
  "Điều 3. Thanh toán",
  "Khoản 3.1"
]
```

### Tờ trình

```json
[
  "TỜ TRÌNH",
  "I. SỰ CẦN THIẾT",
  "1. Bối cảnh thực tế"
]
```

### Quyết định

```json
[
  "QUYẾT ĐỊNH",
  "Điều 1",
  "Khoản 2"
]
```

### Công văn

Có thể là:

```json
[
  "CÔNG VĂN VỀ VIỆC ...",
  "KÍNH GỬI: ...",
  "1. Nội dung đề nghị"
]
```

Không cần official-letter parser riêng.

---

# 26. Node Attachment Rules

## 26.1 Paragraph

Attach vào:

```text
nearest active structural node
```

Nếu chưa có structural node:

```text
DOCUMENT/BODY
```

---

## 26.2 Table

Table fragment attach vào current structural parent.

Ví dụ:

```text
Điều 3
paragraph
TABLE
paragraph
Điều 4
```

TABLE thuộc `Điều 3`.

---

## 26.3 Bullet

Bullet Level 5 chỉ khi hierarchy hợp lý.

Nếu không có parent thích hợp, giữ dưới dạng list-like paragraph thay vì ép orphan Level 5.

---

## 26.4 Closing

`Nơi nhận` không được thuộc Article/Section cuối.

Khi accepted:

```text
close body structural stack
attach RECIPIENTS to document closing
```

Signature tương tự.

---

# 27. Table-Aware Parsing — giữ nguyên và tổng quát hóa SAHC-v2

SAHC-v3 bắt buộc giữ kiến trúc table-aware của v2.

Không embed raw HTML table làm primary representation.

---

## 27.1 ParsedTable

```python
@dataclass
class ParsedTable:
    table_id: str

    page_start: int
    page_end: int

    headers: list[str]
    rows: list[list[str]]

    column_count: int

    source_block_ids: list[str]

    continuation_of: str | None = None
    structural_parent_id: str | None = None

    metadata: dict = field(default_factory=dict)
```

---

## 27.2 HTML parser

```python
def parse_html_table(html: str) -> ParsedTable:
    ...
```

Khuyến nghị:

```text
BeautifulSoup4
```

Hỗ trợ:

```text
thead
tbody
rowspan
colspan
multi-row header
missing thead
malformed-but-recoverable HTML
```

---

# 28. Multi-row Header Flattening

Phải tạo logical schema.

Ví dụ:

```text
Môn học SV đã học
  ├── Mã MH
  ├── Tên môn học
  ├── Số TC
  └── Điểm

Môn học SV được chuyển
  ├── Mã MH chuyển
  ├── Tên môn học
  ├── Số TC được chuyển
  └── Điểm chuyển đổi
```

Generic flattened labels:

```text
Môn học SV đã học > Mã MH
Môn học SV đã học > Tên môn học
Môn học SV đã học > Số TC
Môn học SV đã học > Điểm
Môn học SV được chuyển > Mã MH chuyển
...
```

Domain-specific serializer có thể map sang cleaner labels, nhưng generic schema phải luôn có.

---

# 29. Cross-Page Table Reconstruction

## 29.1 Two-stage architecture

Khuyến nghị:

```text
1. structure_parser tạo TABLE_FRAGMENT nodes.
2. fragment attach vào structural parent hiện tại.
3. table_parser parse schema/rows.
4. continuation detector so các fragments adjacent pages.
5. fragments compatible được merge thành logical table.
```

Cách này tránh circular dependency giữa structure parser và table parser.

---

## 29.2 Continuation signals

Positive:

```text
previous table bbox bottom > 0.80
next table bbox top < 0.20
same/compatible column count
same structural parent
no accepted structural boundary between
same/compatible flattened schema
next page starts with data-like row
```

Negative:

```text
new Article/Section before next table
new Annex
column count strongly differs
new independent table heading
incompatible schema
```

Reference score:

```python
score = 0.0

if prev_bottom > 0.80:
    score += 0.20

if next_top < 0.20:
    score += 0.20

if compatible_column_count:
    score += 0.20

if same_structural_parent:
    score += 0.20

if no_new_boundary_between:
    score += 0.10

if compatible_schema:
    score += 0.20

if independent_table_heading:
    score -= 0.40

if new_structural_boundary:
    score -= 0.60
```

Threshold:

```python
TABLE_CONTINUATION_THRESHOLD = 0.65
```

---

## 29.3 Header inheritance

Nếu page N+1 bắt đầu bằng row bị OCR gắn nhầm thành header:

```text
row looks data-like
+
previous fragment has stable schema
+
continuation score passes
```

thì:

```text
inherit schema page N
treat first row page N+1 as data
```

Không làm mất row.

---

# 30. Table Semantic Serialization

Giữ strategy pattern của v2.

```python
class TableRowSerializer(Protocol):
    def can_handle(self, table: ParsedTable) -> bool:
        ...

    def serialize(
        self,
        table: ParsedTable,
        row: list[str],
        document_meta: dict,
        section_path: list[str],
    ) -> str:
        ...
```

Tối thiểu:

```text
GenericKeyValueTableSerializer
CourseTransferTableSerializer
```

Serializer selection dựa trên table schema/content, không dựa trên document type.

---

## 30.1 Generic serializer

```text
Phần: {section_path}

{Column 1}: {value 1}
{Column 2}: {value 2}
{Column 3}: {value 3}
```

Skip empty values.

Không output:

```text
Column: None
```

---

## 30.2 One semantic row = one AtomicUnit

Default:

```text
one semantic row = one AtomicUnit
```

Đặc biệt với lookup-oriented table.

Chỉ pack nhiều row ngắn nếu:

```text
same logical table
same section_path
token-safe
row boundaries still explicit
```

Default config:

```python
prefer_single_table_row_chunks = True
```

---

# 31. Atomic-Unit Generation từ AST

Walk AST sau khi table reconstruction hoàn tất.

Rules:

```text
legal_basis node
→ legal_basis AtomicUnit

section/article heading có body intro
→ section_intro/article_intro

clause node
→ clause AtomicUnit

point node
→ point AtomicUnit

plain paragraph
→ paragraph AtomicUnit

logical table row
→ table_row AtomicUnit

recipient line
→ optional recipient_item

signature text
→ optional signature
```

Không tạo empty child chỉ vì có heading.

Heading vẫn được giữ trong `section_path`.

---

# 32. Token-Aware Packing

Giữ nguyên nguyên tắc v2:

> Không dùng character count làm token budget.

Không dùng:

```python
len(text)
```

cho quyết định chunk size.

---

## 32.1 TokenCounter

```python
class TokenCounter:
    def __init__(self, model):
        self.model = model
        self.tokenizer = resolve_tokenizer(model)

    def count(self, text: str) -> int:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )
        return len(encoded["input_ids"])
```

`resolve_tokenizer()` phải defensive vì SentenceTransformer version có thể khác nhau.

---

## 32.2 Token budget

Lấy:

```python
max_seq_length = model.max_seq_length
```

Tính:

```python
available_budget = (
    max_seq_length
    - contextual_prefix_tokens
    - special_token_margin
    - safety_margin_tokens
)
```

Final validation phải count toàn bộ `retrieval_text`.

---

## 32.3 Packing compatibility

Chỉ pack atomic units compatible:

```python
def can_pack(a: AtomicUnit, b: AtomicUnit) -> bool:
    return (
        a.parent_id == b.parent_id
        and a.section_path == b.section_path
        and compatible_unit_types(a.unit_type, b.unit_type)
        and not crosses_table_row_boundary_policy(a, b)
    )
```

Không pack hai section khác nhau chỉ vì token còn trống.

---

## 32.4 Long-unit fallback

Nếu một atomic unit vượt budget:

```text
explicit detected descendants
→ paragraph boundaries
→ sentence boundaries
→ token window fallback
```

Ví dụ clause có detected points:

```text
clause
→ point children
```

Plain paragraph dài:

```text
paragraph
→ sentence groups
→ token window
```

Metadata:

```json
{
  "split_fallback": "sentence"
}
```

hoặc:

```json
{
  "split_fallback": "token_window"
}
```

---

## 32.5 Overlap

Không dùng fixed overlap cho toàn bộ chunker.

Contextual overlap đến từ:

```text
document title
metadata
section_path
current heading
table schema
```

Chỉ token-window fallback mới dùng:

```python
fallback_overlap_tokens = 20
```

---

# 33. Parent/Child Retrieval Model

## 33.1 Parent selection

Universal rule:

```text
Parent = nearest context-expansion-worthy structural ancestor
```

Preferred parent types:

```text
article
section
annex subsection
table
preamble
```

Ví dụ table row trong Điều:

```text
parent_id = Article parent
table_parent_id = logical table
```

Tờ trình:

```text
parent_id = nearest Level-1/Level-2 semantic section
```

Công văn ít cấu trúc:

```text
parent_id = nearest body semantic section
```

Không cần document-type logic.

---

## 33.2 Child

Child là vector-search unit.

Mỗi child phải giữ:

```text
parent_id
section_path
page range
source_block_ids
node trace
chunk_type
```

---

## 33.3 Sibling relationship

Thêm:

```text
sibling_group_id
sibling_index
```

Sibling group có thể là:

```text
children cùng structural parent
rows cùng logical table
points cùng clause
```

Cho phép deterministic adjacent-context expansion.

---

# 34. Retrieval Text

## 34.1 Generic builder

```python
def build_retrieval_text(
    document_meta: dict,
    chunk: Chunk,
) -> str:
    ...
```

Format mặc định:

```text
Văn bản: {document title or Summary}
Số: {No}
Cơ quan ban hành: {Author}
Ngày: {DateDocument}
Phần: {section_path joined by " > "}

Nội dung:
{normalized_text}
```

Chỉ include field có giá trị.

---

## 34.2 Prefix budget

Context prefix phải được count token.

Không được xảy ra:

```text
body fits
+
prefix causes overflow
+
model silently truncates
```

Invariant:

```python
token_counter.count(chunk.retrieval_text) <= model.max_seq_length
```

---

## 34.3 Table-row retrieval text

Ví dụ:

```text
Văn bản: HỢP ĐỒNG MUA BÁN ...
Phần: Điều 3. Thanh toán > Bảng tiến độ thanh toán

Nội dung:
Đợt thanh toán: 1
Tỷ lệ: 50%
Thời hạn: Trong vòng 05 ngày...
```

Không prepend toàn bộ parent Article.

---

# 35. Qdrant Payload Schema v3

Child payload đề nghị:

```json
{
  "Id": "...",
  "KeyFileId": "...",
  "Page": "...",
  "No": "...",
  "Author": "...",
  "Summary": "...",
  "DateDocument": "...",
  "RecordId": "...",
  "FileNameMinio": "...",
  "FilePathMinio": "...",

  "document_id": "...",
  "document_title": "HỢP ĐỒNG MUA BÁN",

  "record_type": "child",

  "chunk_id": "...",
  "chunk_index": 17,
  "chunk_type": "clause",

  "node_id": "...",
  "node_type": "clause",
  "node_level": 3,

  "boundary_kind": "decimal_number",
  "boundary_confidence": 0.91,

  "parent_id": "...",
  "table_parent_id": null,

  "sibling_group_id": "...",
  "sibling_index": 2,

  "section_path": [
    "HỢP ĐỒNG MUA BÁN",
    "Điều 3. Thanh toán",
    "3.1. Phương thức thanh toán"
  ],

  "page_start": 2,
  "page_end": 2,

  "raw_text": "...",
  "normalized_text": "...",
  "retrieval_text": "...",

  "token_count": 176,

  "source_block_ids": [
    "page_002_block_011"
  ],

  "table_id": null,
  "table_row_index": null,

  "source": "OCR_JSON",

  "chunking_version": "sahc-v3",
  "parser_version": "3.0.0",
  "grammar_version": "universal-v1"
}
```

---

## 35.1 Parent payload/store

```json
{
  "record_type": "parent",

  "document_id": "...",
  "parent_id": "...",

  "node_id": "...",
  "node_type": "article",

  "section_path": [
    "HỢP ĐỒNG MUA BÁN",
    "Điều 3. Thanh toán"
  ],

  "page_start": 2,
  "page_end": 3,

  "normalized_text": "...",

  "child_ids": [
    "...",
    "..."
  ],

  "source_block_ids": [
    "..."
  ],

  "chunking_version": "sahc-v3"
}
```

Không dùng zero-vector parent trong cùng searchable collection nếu có nguy cơ pollute retrieval.

Ưu tiên:

```text
child vector collection
+
parent store / parent collection
```

hoặc strict `record_type=child` filter.

---

# 36. Retrieval API v3

```python
def embedding_search_v3(
    query: str,
    top_k: int = 10,
    expand_context: bool = True,
):
    ...
```

Flow:

```text
query embedding
→ vector search record_type=child
→ optional parent diversity
→ dedup
→ optional rerank
→ adaptive context expansion
→ retrieval results
```

---

## 36.1 Parent diversity

Config:

```python
max_children_per_parent = 3
```

Tránh top-k bị chiếm bởi các sibling gần giống nhau.

---

## 36.2 Adaptive expansion

Heuristics deterministic:

```text
table-row exact hit:
    return row
    + table schema/title
    + optional nearby rows

general section query:
    expand parent section/article

multiple sibling hits:
    merge sibling context

very long parent:
    do not append full parent blindly
    use relevant sibling window
```

Không cần LLM router.

---

# 37. Stable IDs

Dùng UUID5 deterministic.

Structural node:

```python
node_uuid = uuid.uuid5(
    uuid.UUID(document_id),
    (
        f"sahc-v3:node:"
        f"{boundary_kind}:"
        f"{canonical_numbering_key}:"
        f"{source_signature}"
    )
)
```

Child:

```python
chunk_uuid = uuid.uuid5(
    uuid.UUID(document_id),
    (
        f"sahc-v3:chunk:"
        f"{chunk_type}:"
        f"{canonical_section_path}:"
        f"{source_signature}:"
        f"{local_index}"
    )
)
```

Không chỉ dựa vào global `chunk_index`.

---

# 38. Configuration

```python
@dataclass
class ChunkingConfig:
    # Token safety
    safety_margin_tokens: int = 16
    fallback_overlap_tokens: int = 20

    # Indexing
    index_recipients: bool = True
    index_signature: bool = True

    # Table behavior
    merge_cross_page_tables: bool = True
    prefer_single_table_row_chunks: bool = True
    table_continuation_threshold: float = 0.65

    # Structure thresholds
    explicit_keyword_threshold: float = 0.50
    decimal_threshold: float = 0.65
    roman_threshold: float = 0.70
    primary_number_threshold: float = 0.72
    letter_point_threshold: float = 0.68
    style_heading_threshold: float = 0.75

    # Layout
    header_y_threshold: float = 0.20
    closing_bottom_y_threshold: float = 0.60
    centered_tolerance: float = 0.10

    # Behavior
    enable_style_heading_fallback: bool = True
    enable_v1_txt_fallback: bool = True

    # Versioning
    chunking_version: str = "sahc-v3"
    parser_version: str = "3.0.0"
    grammar_version: str = "universal-v1"
```

Không rải magic number trong production code.

---

# 39. Integration với embedding pipeline

Entry point mới:

```python
def build_document_chunks_v3(
    json_path: str,
    document_meta: dict,
    embedding_model,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    ...
```

Flow:

```text
load OCR JSON
↓
parse OCR blocks
↓
normalize / match-text generation
↓
reading order
↓
logical-span segmentation
↓
layout features
↓
universal structure parser
↓
AST
↓
parse + merge tables
↓
atomic units
↓
token-aware packing
↓
parent/child build
↓
retrieval_text
↓
validate
↓
return chunks
```

Embedding integration:

```python
chunks = build_document_chunks_v3(
    json_path=json_path,
    document_meta=meta,
    embedding_model=model,
    config=config,
)

children = [
    c for c in chunks
    if c.metadata.get("record_type", "child") == "child"
]

for chunk in children:
    assert token_counter.count(chunk.retrieval_text) <= model.max_seq_length

    vector = model.encode(
        chunk.retrieval_text,
        normalize_embeddings=True,
    )
```

Không silent truncation.

---

# 40. Feature Flag và Backward Compatibility

Giữ v1/v2 để benchmark.

Ví dụ:

```python
CHUNKING_VERSION = os.getenv("CHUNKING_VERSION", "v3")

if CHUNKING_VERSION == "v1":
    chunks = chunk_legal_document_v1(...)
elif CHUNKING_VERSION == "v2":
    chunks = build_document_chunks_v2(...)
elif CHUNKING_VERSION == "v3":
    chunks = build_document_chunks_v3(...)
else:
    raise ValueError(...)
```

Không xóa baseline cũ ngay.

---

# 41. Qdrant Collection Safety

Không tự động destroy/recreate production collection.

Đề nghị:

```text
rag_document_v1
rag_document_v2
rag_document_v3
```

hoặc collection có strict version filters.

Config:

```python
COLLECTION_NAME_V3 = "rag_document_v3"
```

Migration phải explicit.

---

# 42. Logging và Audit

Mỗi document log:

```text
document_id
page_count
block_count
logical_span_count

header_blocks
body_blocks
closing_blocks
annex_count

boundary_candidates
accepted_boundaries
rejected_boundaries

level0_count
level1_count
level2_count
level3_count
level4_count
level5_count

article_count
section_count
clause_count
point_count

orphan_boundary_downgraded
false_split_guard_count
style_heading_count

table_fragment_count
logical_table_count
cross_page_table_count

parent_count
child_count

avg_child_tokens
p95_child_tokens
max_child_tokens

sentence_fallback_count
token_window_fallback_count
```

Rejected boundary debug:

```text
[boundary-reject]
span=page_002_block_014_line_02
pattern=article
text="Điều 2 của hợp đồng này..."
reason=sentence_continuation
score=0.31
```

Đây là output quan trọng để tune grammar mà không dùng LLM.

---

# 43. Validators bắt buộc

## 43.1 Token overflow

```python
assert token_counter.count(chunk.retrieval_text) <= model.max_seq_length
```

Nếu false:

```text
ERROR
```

Không upsert.

---

## 43.2 AST level consistency

Với structural node bình thường:

```python
child.level > parent.level
```

trừ các special node:

```text
document
metadata
closing
paragraph
table row
```

---

## 43.3 Parent existence

Mọi child `parent_id` phải resolve.

---

## 43.4 No structural orphan

Không chấp nhận mặc định:

```text
Level 4 directly under DOCUMENT
Level 5 directly under DOCUMENT
```

---

## 43.5 Section path

Embedding child phải có section path nếu document title/structural ancestry tồn tại.

---

## 43.6 Table row validation

Nếu:

```text
chunk_type == table_row
```

thì bắt buộc:

```text
table_id
table_row_index
section_path
structural parent
```

---

## 43.7 Empty chunk

Không upsert empty/near-empty retrieval text.

---

## 43.8 Source traceability

Mọi child phải có ít nhất một:

```text
source_block_id
```

hoặc một trace hợp lệ về source span/block.

---

# 44. Edge-Case Validation Scenarios

Đây là acceptance tests bắt buộc của SAHC-v3.

---

## 44.1 Scenario A — Nested Hợp đồng

Input:

```text
HỢP ĐỒNG MUA BÁN

BÊN A: CÔNG TY ABC
Địa chỉ: ...

BÊN B: CÔNG TY XYZ
Địa chỉ: ...

ĐIỀU 3. THANH TOÁN

3.1. Phương thức thanh toán

a) Bên A thanh toán 50%...
b) Phần còn lại...

3.2. Thời hạn thanh toán

Theo Điều 2 của Hợp đồng này, ...
```

Expected AST:

```text
DOCUMENT "HỢP ĐỒNG MUA BÁN"
├── L1 "BÊN A"
├── L1 "BÊN B"
└── L2 "ĐIỀU 3. THANH TOÁN"
    ├── L3 "3.1. Phương thức thanh toán"
    │   ├── L4 "a) ..."
    │   └── L4 "b) ..."
    └── L3 "3.2. Thời hạn thanh toán"
        └── paragraph "Theo Điều 2..."
```

Critical assertion:

```text
"Điều 2" trong "Theo Điều 2..." KHÔNG tạo Article mới.
```

Expected path:

```json
[
  "HỢP ĐỒNG MUA BÁN",
  "ĐIỀU 3. THANH TOÁN",
  "3.1. Phương thức thanh toán"
]
```

---

## 44.2 Scenario B — Hợp đồng dùng `1.`/`2.` dưới Điều

Input:

```text
ĐIỀU 4. QUYỀN VÀ NGHĨA VỤ

1. Quyền của Bên A
2. Nghĩa vụ của Bên A
```

Expected:

```text
Điều 4      L2
├── 1.      effective L3
└── 2.      effective L3
```

Test này xác nhận primary-number demotion.

---

## 44.3 Scenario C — Tờ trình không có Điều

Input:

```text
TỜ TRÌNH

KÍNH GỬI: BAN GIÁM ĐỐC

I. SỰ CẦN THIẾT

1. Bối cảnh thực tế
Nội dung...

2. Vướng mắc
Nội dung...

II. NỘI DUNG ĐỀ XUẤT

1. Phương án
...
```

Expected:

```text
DOCUMENT "TỜ TRÌNH"
├── L1 "KÍNH GỬI: BAN GIÁM ĐỐC"
├── L1 "I. SỰ CẦN THIẾT"
│   ├── L2 "1. Bối cảnh thực tế"
│   └── L2 "2. Vướng mắc"
└── L1 "II. NỘI DUNG ĐỀ XUẤT"
    └── L2 "1. Phương án"
```

Không document-type logic.

---

## 44.4 Scenario D — Unstructured Tờ trình

Input:

```text
TỜ TRÌNH
Về việc phê duyệt phương án ...

Kính gửi: ...

Căn cứ ...
Căn cứ ...

Nội dung đề xuất như sau:
...
...
Trân trọng kính trình.
```

Expected:

```text
document title
metadata/preamble
legal_basis units
semantic heading nếu confidence đủ
paragraph units
closing nếu có signature/recipient signal
```

Không fail chỉ vì không có Roman/Điều numbering.

---

## 44.5 Scenario E — Công văn

Input:

```text
CÔNG VĂN
V/v cung cấp hồ sơ ...

Kính gửi: Công ty ABC

1. Đề nghị cung cấp ...
2. Thời hạn gửi ...

Nơi nhận:
- Như trên;
- Lưu: VT.

KT. GIÁM ĐỐC
PHÓ GIÁM ĐỐC
Nguyễn Văn A
```

Expected:

```text
DOCUMENT
├── L1 Kính gửi
├── L2 1.
├── L2 2.
└── CLOSING
    ├── RECIPIENTS
    └── SIGNATURE
```

`Nơi nhận` không được là child của `2.`.

---

## 44.6 Scenario F — Quyết định

Input:

```text
QUYẾT ĐỊNH

Căn cứ ...
Căn cứ ...
Theo đề nghị ...

QUYẾT ĐỊNH:

Điều 1. ...
1. ...
a) ...

Điều 2. ...

Nơi nhận:
...
```

Expected:

```text
DOCUMENT
├── PREAMBLE
│   ├── legal_basis
│   ├── legal_basis
│   └── legal_basis
├── article Điều 1
│   ├── effective clause 1.
│   └── point a)
├── article Điều 2
└── closing
```

Standalone `QUYẾT ĐỊNH:` có thể là semantic/style heading nhưng không được phá Article hierarchy.

---

## 44.7 Scenario G — Phụ lục sau chữ ký

Input:

```text
... signature ...

PHỤ LỤC 01
DANH MỤC ...

I. NHÓM A
1. ...
2. ...
```

Expected transition:

```text
CLOSING
→ ANNEX
```

AST:

```text
DOCUMENT
├── CLOSING
└── ANNEX "PHỤ LỤC 01"
    └── L1 "I. NHÓM A"
        ├── L2 "1. ..."
        └── L2 "2. ..."
```

---

## 44.8 Scenario H — Standalone Phụ lục

Input bắt đầu:

```text
PHỤ LỤC
DANH SÁCH ...
```

Expected:

```text
accept annex/root section immediately
```

Không cần parent primary document.

---

## 44.9 Scenario I — Cross-page table cần merge

Page 1:

```text
Điều 1. ...
[TABLE fragment A near page bottom]
```

Page 2:

```text
[TABLE fragment B near page top]
paragraph
Điều 2. ...
```

Expected:

```text
one logical table
page_start = 1
page_end = 2
same structural_parent_id = Điều 1
```

Rows page 2 kế thừa schema page 1 nếu phù hợp.

---

## 44.10 Scenario J — Table không được merge

Page 1:

```text
TABLE A
```

Page 2:

```text
Điều 2. ...
TABLE B
```

Expected:

```text
TABLE A != TABLE B
```

New structural boundary là strong negative signal.

---

## 44.11 Scenario K — OCR line-wrap false Article

Một OCR block:

```text
Bên A thực hiện nghĩa vụ theo
Điều 2 của Hợp đồng này.
```

Expected:

```text
one paragraph
```

Không Article mới.

---

## 44.12 Scenario L — OCR spaced heading

Input:

```text
Đ I Ề U  5 .  B Ả O  M Ậ T
```

Nếu prefix compaction + layout evidence đủ mạnh:

```text
accept Article 5
```

nhưng raw text phải giữ nguyên.

---

## 44.13 Scenario M — Ambiguous numbered list

Input:

```text
Các hồ sơ gồm:
1. Bản sao CCCD;
2. Giấy đề nghị;
3. Tài liệu liên quan.
```

Nếu list nằm trong paragraph-like section và layout không giống heading, preferred representation:

```text
list items / shallow child units under current section
```

Không preferred:

```text
new peer Level-2 sections làm đóng active semantic parent
```

Resolver dùng:

```text
active stack
sequence
indentation
short list-item style
punctuation
```

---

## 44.14 Scenario N — Mid-sentence `Khoản 1`

Input:

```text
Bên B thực hiện nghĩa vụ theo Khoản 1 Điều 5 của Hợp đồng.
```

Expected:

```text
one paragraph
```

Không tạo Clause.

---

## 44.15 Scenario O — Number/date false positive

Input:

```text
Giá trị hợp đồng: 1.500.000.000 đồng.
Ngày hiệu lực: 01.08.2026.
Tỷ lệ phạt: 2.5%.
```

Expected:

```text
no hierarchy boundaries
```

---

# 45. Unit Test Skeletons

## 45.1 False Article split

```python
def test_inline_article_reference_is_not_boundary():
    text = "Theo Điều 2 của Hợp đồng này, Bên A có trách nhiệm ..."

    span = make_test_span(
        text=text,
        paragraph_start=True,
    )

    candidate = detect_best_boundary_candidate(...)

    assert candidate is None or candidate.kind != "article"
```

---

## 45.2 Effective Level under Article

```python
def test_primary_number_is_demoted_under_article():
    stack = [
        node("document", level=None),
        node("article", level=2, title="Điều 4"),
    ]

    candidate = BoundaryCandidate(
        kind="primary_number",
        nominal_level=2,
        effective_level=None,
        numbering_key="1",
        ...
    )

    level = resolve_effective_level(candidate, stack, None)

    assert level == 3
```

---

## 45.3 Tờ trình hierarchy

```python
def test_roman_then_primary_number():
    spans = [
        "I. SỰ CẦN THIẾT",
        "1. Bối cảnh thực tế",
        "2. Vướng mắc",
        "II. NỘI DUNG ĐỀ XUẤT",
        "1. Phương án",
    ]

    ast = parse_text_fixture(spans)

    assert_path(
        ast,
        ["I. SỰ CẦN THIẾT", "1. Bối cảnh thực tế"],
    )
```

---

## 45.4 Cross-page table

```python
def test_cross_page_table_keeps_structural_parent():
    ast, tables = parse_fixture("cross_page_table.json")

    assert len(tables) == 1

    table = tables[0]

    assert table.page_start == 1
    assert table.page_end == 2
    assert table.structural_parent_id == find_node_id(ast, "Điều 1")
```

---

# 46. Debug CLI

Giữ và mở rộng debug CLI của v2.

```bash
python -m chunking.debug \
  --input path/to/file.json \
  --meta path/to/meta.json \
  --version v3 \
  --output chunks_debug.json \
  --output-markdown chunks_debug.md
```

JSON output:

```json
{
  "document": {},
  "parser_stats": {},
  "boundary_decisions": [],
  "ast": {},
  "tables": [],
  "parents": [],
  "children": []
}
```

Markdown debug nên hiển thị:

```markdown
# Document

## AST

- HỢP ĐỒNG MUA BÁN
  - Điều 3. Thanh toán
    - 3.1. Phương thức

## Boundary decisions

### Accepted
...

### Rejected
...

## Parent: Điều 3. Thanh toán

### Child 1 — clause
...
```

Boundary reject report rất quan trọng để cải thiện grammar.

---

# 47. Evaluation Dataset cho SAHC-v3

Không chỉ benchmark Quyết định.

Tối thiểu cần tập gồm:

```text
3+ Quyết định
3+ Hợp đồng
3+ Tờ trình
3+ Công văn
3+ Phụ lục / tài liệu có annex
```

Nếu dữ liệu chưa đủ, dùng fixture synthetic cho parser test nhưng retrieval benchmark chính phải dùng tài liệu thật khi có thể.

Query categories:

```text
metadata lookup
section/article lookup
clause lookup
point lookup
cross-reference query
table-row lookup
cross-page table lookup
annex lookup
entity-specific lookup
```

---

# 48. Evaluation Metrics

Giữ retrieval metrics của v2:

```text
Recall@1
Recall@3
Recall@5
MRR
nDCG@5 nếu có graded relevance
```

Bổ sung structure-specific metrics cho v3:

```text
Boundary Precision
Boundary Recall
Boundary F1
Hierarchy Parent Accuracy
Section-Path Exact Match
False-Split Rate
Missed-Boundary Rate
Table Continuation Accuracy
```

Engineering metrics:

```text
avg child tokens
p95 child tokens
max child tokens
child count/document
index size
retrieval latency
parser latency/document
token-window fallback rate
```

---

## 48.1 Boundary metrics

Gold annotation tối thiểu:

```text
span_id
boundary_kind
level
parent_boundary_id
```

Tính:

```text
Boundary Precision = correct_detected / all_detected
Boundary Recall = correct_detected / all_gold
Boundary F1 = harmonic mean
```

---

## 48.2 Hierarchy Parent Accuracy

Cho mỗi gold structural node:

```text
predicted parent == gold parent
```

Metric này quan trọng hơn chỉ detect heading vì parser có thể detect đúng boundary nhưng attach sai parent.

---

## 48.3 Section-Path Exact Match

Ví dụ gold:

```json
[
  "TỜ TRÌNH",
  "I. SỰ CẦN THIẾT",
  "1. Bối cảnh thực tế"
]
```

Predicted phải match sau canonical whitespace normalization.

---

## 48.4 False-Split Rate

Tạo tập hard negatives chứa:

```text
theo Điều 2...
tại Khoản 1...
ngày 01.08.2026
1.500.000 đồng
OCR line-wrap trước Điều
```

Metric:

```text
False-Split Rate = false boundaries / negative candidates
```

---

# 49. Fair Baseline Comparison

Để đo contribution của chunker, giữ cố định:

```text
embedding model
embedding normalization
Qdrant distance
query set
top_k
reranker setting
retrieval filters
```

Chỉ thay chunking strategy.

Recommended comparison:

```text
Fixed-size token chunking
Recursive character/token splitter
Semantic chunking baseline
SAHC-v2
SAHC-v3
```

Nếu semantic chunking dùng embedding/model riêng để quyết định boundary, phải report rõ additional compute và dependency.

Mục tiêu benchmark v3 là chứng minh gain đến từ:

```text
universal structural grammar
hierarchy preservation
table handling
parent/child retrieval
token-safe packing
```

---

# 50. Implementation Priority cho Codex

## Phase 1 — Foundation

```text
1. config.py
2. models.py extensions
3. normalize.py + match_text
4. reading_order.py
5. span_segmenter.py
6. layout_features.py
```

---

## Phase 2 — Universal Grammar

```text
7. boundary_registry.py
8. regex matchers
9. semantic heading registry
10. boundary_scoring.py
11. false-split guards
12. numbering sequence helpers
```

---

## Phase 3 — State Machine

```text
13. ParserState
14. effective-level resolver
15. stack pop/push rules
16. zone transitions
17. generic node creation
18. generic section_path
19. structure tests
```

---

## Phase 4 — Tables

```text
20. preserve v2 HTML parser
21. multi-row header flattening
22. table fragments attached to AST
23. cross-page continuation
24. semantic row serialization
25. cross-page tests
```

---

## Phase 5 — Chunk Creation

```text
26. AST → AtomicUnit
27. token-aware packing
28. parent selection
29. sibling groups
30. child creation
31. retrieval text
32. validators
```

---

## Phase 6 — Integration

```text
33. embedding_v3 integration
34. Qdrant v3 payload
35. new collection/config
36. search_v3
37. parent/sibling expansion
38. debug outputs
```

---

## Phase 7 — Evaluation

```text
39. parser gold fixtures
40. boundary metrics
41. multi-document query set
42. v1/v2/v3 retrieval benchmark
43. error analysis
```

---

# 51. Coding Rules cho Codex

1. Code type-hinted.
2. Tách pure functions tối đa có thể.
3. Không hard-code document IDs mẫu trong production code.
4. Không dùng LLM API trong chunking.
5. Không thêm document classifier.
6. Không có `if doc_type == ...`.
7. Heuristic phải nằm trong named function rõ ràng.
8. Mọi heuristic cần test.
9. Không sửa `raw_text`.
10. Không silent fallback khi parser lỗi nghiêm trọng.
11. Log warning khi table parse fail.
12. Log uncertain cross-page continuation.
13. Log token-window fallback.
14. Log downgraded orphan boundary.
15. Ưu tiên deterministic output.
16. Mỗi structural node phải trace về source block/span.
17. Không silently truncate embedding text.
18. Không embed raw HTML table làm representation chính.
19. Không rải regex/magic threshold trong `structure_parser.py`; đưa registry/config vào module riêng.
20. Không để style-heading fallback override explicit grammar.

---

# 52. Acceptance Criteria

Implementation chỉ hoàn tất khi thỏa tất cả.

## 52.1 Universal structure

- [ ] Cùng một parser xử lý được Quyết định, Tờ trình, Hợp đồng, Công văn và Phụ lục.
- [ ] Không có document classifier.
- [ ] Không có `if doc_type == ...` trong structure parsing.
- [ ] Parse được `PHẦN`, `CHƯƠNG`, `MỤC`, Roman, `Điều`, `Khoản`, decimal numbering, `Điểm`, letter points, bullet.
- [ ] Generic numbering có nominal/effective level.
- [ ] `1.` dưới `Điều` được demote hợp lý.
- [ ] `1.` dưới `I.` có thể giữ Level 2.
- [ ] Không tạo orphan Level 4/5 mặc định.

## 52.2 False-split safety

- [ ] `Theo Điều 2...` không mở Article.
- [ ] `tại Khoản 1...` không mở Clause.
- [ ] OCR line-wrap trước `Điều` không tạo false boundary khi là sentence continuation.
- [ ] Date/amount/percentage không bị parse thành hierarchy.
- [ ] Table cell numbering không mở document node.

## 52.3 Header/closing/annex

- [ ] Header zone sử dụng bbox nhưng không nuốt document title.
- [ ] `Nơi nhận` không thuộc section cuối.
- [ ] Signature có closing node riêng khi confidence đủ.
- [ ] `PHỤ LỤC` sau signature mở ANNEX subtree mới.
- [ ] Standalone appendix vẫn parse được.

## 52.4 Section path

- [ ] Path được build từ AST ancestry.
- [ ] Hợp đồng path hoạt động.
- [ ] Tờ trình path hoạt động.
- [ ] Quyết định path hoạt động.
- [ ] Không có doc-specific path builder.

## 52.5 Table

- [ ] Parse HTML table.
- [ ] Flatten multi-row headers.
- [ ] Cross-page table merge hoạt động.
- [ ] Page N+1 kế thừa schema khi phù hợp.
- [ ] Table fragments biết structural parent.
- [ ] Table row được serialize key-value.
- [ ] Không embed raw HTML table làm primary child.

## 52.6 Token

- [ ] Dùng tokenizer thật của embedding model.
- [ ] Prefix được count trong budget.
- [ ] Không child nào vượt `model.max_seq_length`.
- [ ] Không dùng `len(text)` làm core chunk budget.
- [ ] Không silent truncation.

## 52.7 Parent/Child

- [ ] Mọi child có `parent_id` hợp lệ.
- [ ] Table row biết Article/Section parent.
- [ ] Có `sibling_group_id` hoặc equivalent.
- [ ] Retrieval có parent/sibling expansion.

## 52.8 Storage/versioning

- [ ] Payload có `section_path`.
- [ ] Payload có `node_type`/`node_level`.
- [ ] Payload có `boundary_kind`/confidence khi applicable.
- [ ] Payload có page range.
- [ ] Payload có raw/normalized/retrieval text.
- [ ] Payload có source trace.
- [ ] Payload có `chunking_version=sahc-v3`.
- [ ] Không destroy production collection tự động.

---

# 53. Non-Goals của SAHC-v3

Không nằm trong scope implementation đầu tiên:

```text
LLM-based boundary detection
LLM OCR correction
query-dependent chunking
Late Chunking
Mixture-of-Chunkers
multimodal image embedding
graph RAG
semantic document classifier
agentic parser
```

SAHC-v3 ưu tiên:

```text
deterministic
auditable
reproducible
document-agnostic
layout-aware
hierarchy-aware
table-aware
token-safe
```

---

# 54. Definition of Done

Khi Codex hoàn tất implementation, phải trả:

```text
1. Danh sách file tạo/sửa
2. Kiến trúc implementation thực tế
3. Boundary registry đã triển khai
4. Effective-level resolver đã triển khai
5. False-split guards
6. Test command
7. Test results
8. AST debug example cho nhiều document styles
9. section_path examples
10. cross-page table example
11. table-row serialization example
12. token statistics
13. Qdrant payload example
14. retrieval expansion example
15. known limitations
16. follow-up recommendations
```

Không chỉ trả:

```text
Implemented successfully
```

mà không có evidence.

---

# 55. Prompt hành động cuối cùng cho Codex

Thực hiện implementation dựa trên toàn bộ đặc tả này.

Trước khi sửa code:

1. Đọc implementation SAHC-v2 hiện tại.
2. Đọc `embedding.py` / `embbeding.py` / `embedding_v1.py` tương ứng trong repo.
3. Xác định cấu trúc repository.
4. Xác định schema OCR JSON thật.
5. Xác định style fields nào thật sự có trong OCR JSON.
6. Giữ backward compatibility nếu hợp lý.
7. Giữ baseline v1/v2 để benchmark.

Sau đó triển khai theo phases ở trên.

Nếu implementation thực tế khác với spec:

```text
what changed
why
impact
follow-up
```

Nhưng không được bỏ các invariant:

```text
NO document classifier
NO document-type branching
NO LLM dependency
Structure-Aware
Table-Aware
Parent/Child
Token-Aware
No silent truncation
Raw OCR preserved
```

---

# 56. Kiến trúc cuối cùng kỳ vọng

```text
                  OCR JSON
                     │
                     ▼
              OCRBlock parsing
                     │
                     ▼
        normalize + parser match text
                     │
                     ▼
              reading order
                     │
                     ▼
              LogicalSpan[]
                     │
                     ▼
             layout features
                     │
                     ▼
          universal zone signals
                     │
                     ▼
         universal boundary registry
                     │
                     ▼
         scoring + false-split guards
                     │
                     ▼
          effective-level resolver
                     │
                     ▼
          deterministic stack AST
                     │
           ┌─────────┴─────────┐
           │                   │
           ▼                   ▼
       text nodes        table fragments
           │                   │
           │                   ▼
           │            HTML table parser
           │                   │
           │             schema flattening
           │                   │
           │            cross-page merging
           │                   │
           │            semantic row units
           │                   │
           └─────────┬─────────┘
                     ▼
                AtomicUnit[]
                     │
                     ▼
             token-aware packing
                     │
                     ▼
          parent / child generation
                     │
                     ▼
             retrieval_text
                     │
                     ▼
            tokenizer validation
                     │
                     ▼
                 embedding
                     │
                     ▼
               Qdrant V3
                     │
                     ▼
              child retrieval
                     │
                     ▼
          parent/sibling expansion
                     │
                     ▼
                   RAG
```

---

# 57. Nguyên tắc quan trọng nhất

Nếu chỉ nhớ một nguyên tắc của SAHC-v3:

> **Không hỏi “đây là loại văn bản gì?” trước khi chunk. Hãy hỏi “span này là boundary gì, level hiệu dụng của nó là bao nhiêu, và nó thuộc ancestor nào trong grammar hiện tại?”**

Đối với cấu trúc:

> **Explicit grammar cung cấp nominal hierarchy; parser state và layout resolve hierarchy hiệu dụng khi numbering ambiguous.**

Đối với false split:

> **Một từ như `Điều`, `Khoản`, `Điểm` chỉ là structural boundary khi xuất hiện ở structural start với đủ evidence; trích dẫn nội tuyến không được làm thay đổi AST.**

Đối với bảng:

> **Table fragment phải giữ structural parent, cross-page fragments phải được reconstruct trước khi serialize từng semantic row.**

Đối với retrieval:

> **Child nhỏ để tìm chính xác; parent/siblings cung cấp context rộng hơn khi cần.**

Đối với tokenizer:

> **Không một `retrieval_text` nào được vượt embedding model context và bị truncate âm thầm.**

---

# 58. Tên version đề xuất

```text
SAHC-v3
```

Full architecture:

```text
SAHC-v3 =
Universal Structure-Aware Hierarchical Chunking
+ Deterministic Boundary Registry
+ Contextual Effective-Level Resolution
+ Table-Aware Row Chunking
+ Cross-Page Table Reconstruction
+ Parent/Child Retrieval
+ Token-Aware Packing
```

Đây là candidate nên dùng để benchmark với:

```text
V1 = Điều regex baseline
V2 = Decision-oriented SAHC
V3 = Universal Multi-Document SAHC
```

---

# 59. Tóm tắt các thay đổi trực tiếp từ SAHC-v2 sang SAHC-v3

| Thành phần | SAHC-v2 | SAHC-v3 |
|---|---|---|
| Input chính | OCR JSON | OCR JSON |
| Structure parser | thiên về Quyết định/Điều | universal grammar |
| Document classifier | không cần | **cấm dùng** |
| Boundary | Điều/Khoản/Điểm + special blocks | registry Level 0–5 |
| Generic numbering | hạn chế | nominal + effective level |
| Layout fallback | cơ bản | bbox/style/heading scoring |
| False-split | start-of-block guard | start + citation + continuation + numeric guards |
| Section path | chủ yếu decision structure | ancestry-based universal path |
| Header | metadata | zone-aware metadata/title distinction |
| Closing | recipients/signature | closing zone + annex re-entry |
| Phụ lục | chưa là first-class universal branch | first-class ANNEX subtree |
| Table-aware | có | giữ nguyên + attach table fragments vào AST |
| Cross-page table | có | giữ nguyên + structural-parent signal |
| Token-aware | có | giữ nguyên |
| Parent/Child | có | giữ nguyên + sibling grouping |
| Qdrant payload | v2 fields | universal structural trace |
| Debug | chunks/tables | thêm accepted/rejected boundary decisions |
| Evaluation | retrieval-centric | retrieval + boundary/hierarchy metrics |

---

# 60. Source-of-Truth Rule

Sau khi được chấp thuận, file này phải trở thành source of truth cho implementation:

```text
CHUNKING_V3_UNIVERSAL_SPEC.md
```

Nếu một module cũ của v2 xung đột với nguyên lý universal parser trong file này, ưu tiên behavior v3 ở các điểm:

```text
universal boundary registry
nominal/effective level separation
no document-type branching
false-split prevention
AST ancestry-based section_path
annex as structural subtree
```

Các capability v2 sau vẫn bắt buộc giữ:

```text
Table-Aware Row Chunking
Cross-Page Table Reconstruction
Token-Aware Packing
Parent/Child Retrieval
Raw OCR preservation
No silent truncation
Deterministic IDs/versioning
```
