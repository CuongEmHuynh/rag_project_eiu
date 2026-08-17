# CHUNKING V2 — IMPLEMENTATION SPECIFICATION FOR CODEX

## 0. Mục tiêu tài liệu

Tài liệu này là đặc tả triển khai để Codex xây dựng phương pháp chunking mới cho hệ thống RAG xử lý văn bản hành chính/doanh nghiệp OCR.

Phương pháp cần triển khai:

> **Structure-Aware Hierarchical Chunking + Table-Aware Row Chunking + Parent/Child Retrieval + Token-Aware Packing**

Mục tiêu không phải chỉ thay một regex chunking. Mục tiêu là thay representation hiện tại:

```text
OCR TXT
→ clean
→ split Điều bằng regex
→ embedding
→ Qdrant
```

bằng pipeline:

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
→ store in Qdrant
→ retrieve children
→ expand parent/siblings when needed
```

Tài liệu này phải được xem là **source of truth cho implementation v2**.

---

# 1. Bối cảnh code hiện tại

File code hiện tại được cung cấp là:

```text
embbeding.py
```

Nếu repository thực tế đã đổi tên thành:

```text
embedding_v1.py
```

thì áp dụng các yêu cầu trong tài liệu này lên file tương ứng.

Code hiện tại có các thành phần chính:

```python
MODEL_EMBEDDING = "bkai-foundation-models/vietnamese-bi-encoder"

def chunk_legal_document(text: str):
    parts = re.split(r"(?=Điều\s+\d+\s*:)", text)
    ...
    if len(p) > 3000:
        ...
```

Và hiện tại đọc OCR từ:

```python
ocr_path = f"./data/file_contents/{META['Id']}.txt"
```

Sau đó:

```python
ocr_text_cleaned = clean_data(...)
chunks = chunk_legal_document(ocr_text_cleaned)
```

### Vấn đề chính của implementation hiện tại

1. Chunk từ `.txt` nên mất phần lớn thông tin cấu trúc có trong OCR JSON.
2. Chỉ hiểu boundary dạng `Điều x:`.
3. Không hiểu:
   - Căn cứ pháp lý;
   - Khoản;
   - Điểm;
   - Bảng;
   - Bảng tiếp tục qua nhiều trang;
   - Nơi nhận;
   - Chữ ký;
   - Figure/stamp noise.
4. Dùng `len(text) > 3000`, tức character count, không phải tokenizer thật của embedding model.
5. Không có parent/child relationship.
6. Không có `section_path`.
7. Không có table row semantic serialization.
8. Raw HTML table có nguy cơ được embedding như một text block lớn.
9. Các block OCR nhiễu có thể lọt vào semantic index.
10. Retrieval hiện tại chỉ lấy top-k vector trực tiếp, không context expansion.

---

# 2. Dữ liệu OCR đầu vào

JSON OCR hiện tại có cấu trúc gần như:

```json
{
  "input_file": "...pdf",
  "page_count": 2,
  "pages": [
    {
      "page_number": 1,
      "blocks": [
        {
          "type": "title",
          "bbox": [0.4, 0.1, 0.6, 0.2],
          "angle": 0,
          "content": "QUYẾT ĐỊNH"
        },
        {
          "type": "text",
          "bbox": [...],
          "content": "Điều 1: ..."
        },
        {
          "type": "table",
          "bbox": [...],
          "content": "<table>...</table>"
        }
      ]
    }
  ]
}
```

Các loại block đã xuất hiện trong dữ liệu mẫu:

```text
title
text
table
table_footnote
figure
figure_caption
abandon
```

Các field có giá trị cần giữ:

```text
page_number
type
bbox
content
angle
```

## Quy tắc bắt buộc

**JSON OCR phải trở thành nguồn input chính của chunker.**

TXT chỉ được dùng cho:

- fallback;
- debug;
- backward compatibility;
- comparison với baseline.

Không dùng TXT làm nguồn production chính cho Chunking V2.

---

# 3. Kiến trúc module cần tạo

Tạo package mới:

```text
chunking/
├── __init__.py
├── models.py
├── normalize.py
├── ocr_parser.py
├── structure_parser.py
├── table_parser.py
├── token_counter.py
├── token_packer.py
├── chunk_builder.py
├── retrieval_text.py
└── validators.py
```

Có thể bổ sung:

```text
tests/
├── test_structure_parser.py
├── test_table_parser.py
├── test_token_packer.py
├── test_chunk_builder.py
├── test_cross_page_table.py
└── test_end_to_end_chunking.py
```

Không đặt toàn bộ logic mới vào một file `embbeding.py`.

`embbeding.py`/`embedding_v1.py` sau refactor chỉ nên chịu trách nhiệm:

```text
load metadata
→ call chunking pipeline
→ encode retrieval_text
→ upsert Qdrant
```

---

# 4. Data model

Sử dụng `dataclass` hoặc Pydantic.

## 4.1 OCRBlock

```python
@dataclass
class OCRBlock:
    page_number: int
    block_index: int
    block_type: str
    bbox: tuple[float, float, float, float] | None
    content_raw: str
    content_normalized: str
    angle: float | int | None = None
```

### Yêu cầu

`content_raw`:

- giữ nguyên OCR;
- không sửa;
- dùng cho audit/citation/debug.

`content_normalized`:

- dùng cho parsing/retrieval;
- chỉ normalize nhẹ;
- không được tự suy diễn hay sửa nội dung pháp lý.

---

## 4.2 DocumentNode

```python
@dataclass
class DocumentNode:
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

    metadata: dict
```

Các `node_type` tối thiểu:

```text
document
metadata
preamble
legal_basis
decision_heading
article
clause
point
paragraph
table
table_row
recipients
signature
other
```

---

## 4.3 Chunk

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

---

# 5. Pipeline tổng thể

Tạo entry point:

```python
def build_document_chunks(
    json_path: str,
    document_meta: dict,
    embedding_model,
) -> list[Chunk]:
    ...
```

Flow bắt buộc:

```text
load OCR JSON
↓
parse blocks
↓
normalize
↓
filter obvious noise
↓
sort reading order
↓
detect document structure
↓
reconstruct cross-page tables
↓
build document tree
↓
create atomic units
↓
token-aware packing
↓
build parent chunks
↓
build child chunks
↓
build retrieval_text
↓
validate
↓
return chunks
```

---

# 6. Normalization

## 6.1 Mục tiêu

Normalize để parser hoạt động ổn định hơn nhưng không phá raw OCR.

Tạo:

```python
def normalize_ocr_text(text: str) -> str:
    ...
```

### Cho phép

- Unicode normalize;
- NBSP → space;
- collapse repeated horizontal whitespace;
- collapse quá nhiều blank line;
- trim đầu/cuối;
- nối line-break OCR khi rất chắc chắn;
- normalize các label layout nội bộ nếu cần.

### Không được tự động

Không tự sửa:

```text
?
8:
D.
C%
A.
```

thành nội dung mà hệ thống "đoán" là đúng.

Không tự sửa:

```text
Kỳ thuật → Kỹ thuật
chuyến → chuyển
```

trừ khi có module OCR correction riêng và được bật explicit.

Giữ:

```text
raw_text
normalized_text
```

song song.

---

# 7. Noise filtering

Các block sau mặc định không tạo semantic child chunk:

```text
page number
table_footnote kiểu tên người xử lý
empty abandon
stamp noise
figure OCR không đủ semantic
```

Ví dụ loại noise:

```text
53
71
72
Quỳnh (ĐT)
```

## 7.1 Không xóa vật lý khỏi document

Noise có thể:

- giữ trong raw document;
- đánh dấu `is_indexable=False`.

Tạo hàm:

```python
def is_indexable_block(block: OCRBlock) -> bool:
    ...
```

Không hard-code chỉ dựa vào `block_type="abandon"` vì một số metadata có giá trị có thể bị OCR model gán `abandon`, ví dụ:

```text
Số: ...
BỘ GIÁO DỤC VÀ ĐÀO TẠO
```

Do đó cần kết hợp:

- block type;
- content;
- bbox;
- lexical patterns;
- position;
- length.

---

# 8. Reading order

Dùng:

```text
page_number
then bbox.y1
then bbox.x1
```

làm baseline.

Tuy nhiên phần header hai cột đầu trang có thể có thứ tự trái/phải khác nhau.

Không cần giải quyết document layout tổng quát hoàn hảo ở v2, nhưng phải đảm bảo:

```text
title / subject / preamble / Điều / table / next Điều
```

không bị đảo.

Tạo:

```python
def sort_blocks_in_reading_order(blocks: list[OCRBlock]) -> list[OCRBlock]:
    ...
```

---

# 9. Structure-Aware Hierarchical Parsing

## 9.1 Mục tiêu

Xây document tree thay vì flat text.

Ví dụ:

```text
DOCUMENT
├── METADATA
├── PREAMBLE
│   ├── LEGAL_BASIS
│   ├── LEGAL_BASIS
│   └── LEGAL_BASIS
├── DECISION
│   ├── ARTICLE 1
│   │   ├── PARAGRAPH
│   │   ├── TABLE
│   │   └── PARAGRAPH
│   ├── ARTICLE 2
│   └── ARTICLE 3
├── RECIPIENTS
└── SIGNATURE
```

---

# 10. Detect document boundaries

Cần regex tolerant với OCR.

## 10.1 Điều

Phải nhận:

```text
Điều 1:
Điều 1.
Điều 1
ĐIỀU 1:
```

Regex gợi ý:

```python
ARTICLE_RE = re.compile(
    r"^\s*Điều\s+(\d+[A-Za-z]?)\s*[:.\-]?\s*",
    re.IGNORECASE
)
```

Không split nếu từ "điều 1" chỉ xuất hiện giữa câu:

```text
Sinh viên có tên ở điều 1 phải...
```

Chỉ detect article boundary khi pattern nằm ở **đầu block/đầu paragraph**.

---

## 10.2 Khoản

Có thể nhận:

```text
1.
2.
3.
```

nhưng chỉ khi đang ở trong Điều và pattern đủ tin cậy.

Không được áp dụng regex số thứ tự toàn tài liệu một cách mù quáng.

---

## 10.3 Điểm

Có thể nhận:

```text
a)
b)
c)
```

hoặc:

```text
a.
b.
```

chỉ khi context cho thấy đang ở Khoản/Điều.

---

# 11. Phần căn cứ

Preamble thường gồm:

```text
- Căn cứ ...
- Căn cứ ...
Căn cứ ...
- Theo đề nghị ...
```

Tách mỗi item thành atomic semantic unit.

Regex phải tolerant OCR:

```text
Căn cứ
Căn cử
Căn cứ vào
Theo đề nghị
```

Không bắt buộc sửa OCR thành tiếng Việt chuẩn.

Mỗi legal basis:

```python
DocumentNode(
    node_type="legal_basis",
    ...
)
```

Sau đó Token-Aware Packing có thể ghép nhiều legal basis nếu còn budget.

---

# 12. Điều + block tiếp theo

Nếu gặp:

```text
Điều 1: ...
TABLE
paragraph
paragraph
Điều 2: ...
```

thì table và các paragraph trước Điều 2 phải thuộc Parent `Điều 1`.

Rule:

```text
current_article = Article 1

mọi block sau Article 1
→ attach vào Article 1
cho tới khi:
    Article mới
    hoặc recipients/signature boundary
```

---

# 13. Detect "Nơi nhận"

Regex tolerant:

```python
RECIPIENT_RE = re.compile(
    r"^\s*Nơi\s+nhận\s*[:.]?",
    re.IGNORECASE
)
```

Sau `Nơi nhận`, các block recipient gần đó gán vào:

```text
RECIPIENTS
```

Không mặc định dùng recipients làm primary embedding chunk nếu không cần.

Cho phép cấu hình:

```python
INDEX_RECIPIENTS = True
```

---

# 14. Signature

Signature có thể đến từ:

```text
figure
figure_caption
text
```

Ví dụ tên người ký có thể OCR đúng ở `figure_caption`.

Tạo parent:

```text
SIGNATURE
```

Nhưng không embedding toàn bộ figure OCR nhiễu.

Có thể index:

```text
chức vụ
tên người ký
```

nếu đủ confidence.

---

# 15. Table-Aware Parsing

Đây là phần bắt buộc, không được bỏ qua.

OCR table đang ở dạng HTML:

```html
<table>
  <thead>...</thead>
  <tbody>
    <tr>...</tr>
  </tbody>
</table>
```

Tạo:

```python
def parse_html_table(html: str) -> ParsedTable:
    ...
```

Khuyến nghị dùng:

```text
BeautifulSoup4
```

hoặc parser HTML ổn định tương đương.

---

# 16. ParsedTable model

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

    metadata: dict = field(default_factory=dict)
```

---

# 17. Multi-row / merged headers

Bảng mẫu có:

```text
rowspan
colspan
multi-row header
```

Không được giả định:

```text
<thead><tr> = header cuối cùng
```

Cần flatten header thành semantic column labels.

Ví dụ bảng chuyển điểm:

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

Kết quả schema nên gần:

```python
[
    "source_course_code",
    "source_course_name",
    "source_credits",
    "source_grade",
    "target_course_code",
    "target_course_name",
    "target_credits",
    "target_grade",
]
```

Không cần map được 100% mọi table trong v2.

Nhưng phải hỗ trợ tốt mẫu "course transfer table".

---

# 18. Cross-page Table Reconstruction

## 18.1 Vấn đề

Bảng có thể bắt đầu ở cuối trang N và tiếp tục ở đầu trang N+1.

Ví dụ:

```text
Page 1
Article 1
Table part A

Page 2
Table part B
Paragraph
Article 2
```

Nếu không merge:

- phần trang 2 mất header;
- row semantics sai;
- table_id bị chia;
- retrieval mất context.

---

# 19. Heuristic detect continuation

Tạo:

```python
def is_table_continuation(
    previous_table: ParsedTable,
    next_table: ParsedTable,
    previous_page_blocks: list[OCRBlock],
    next_page_blocks: list[OCRBlock],
) -> bool:
    ...
```

Điểm số heuristic:

### Positive signals

1. Previous table ở gần cuối page:
   ```text
   bbox bottom > 0.80
   ```

2. Next table ở gần đầu page:
   ```text
   bbox top < 0.20
   ```

3. Column count giống hoặc gần giống.

4. Next page trước table không có:
   ```text
   new title
   new Article
   recipients
   ```

5. Current structural parent vẫn là cùng Điều.

6. Next table không có một header semantic mới rõ ràng.

### Negative signals

1. Có Điều mới trước next table.
2. Có title mới.
3. Column count khác mạnh.
4. Table schema khác hoàn toàn.

---

# 20. Merge table continuation

Nếu continuation:

```python
merged_table.page_end = next_table.page_end
merged_table.rows.extend(next_table.rows)
merged_table.source_block_ids.extend(next_table.source_block_ids)
```

Quan trọng:

Nếu table trang 2 không có header đúng mà parser coi row đầu là `<thead>`, cần có logic:

```text
nếu row đầu của page 2 giống data row
→ không xem là schema mới
→ dùng schema của page 1
```

---

# 21. Table semantic serialization

Không embed raw HTML làm representation chính.

Mỗi row phải serialize thành key-value semantic text.

Tạo interface:

```python
def serialize_table_row(
    table: ParsedTable,
    row: list[str],
    document_meta: dict,
    section_path: list[str],
) -> str:
    ...
```

Ví dụ:

```text
Văn bản: Quyết định chuyển điểm cho sinh viên Võ Hoàng Duy
Phần: QUYẾT ĐỊNH > Điều 1 > Bảng chuyển điểm

Môn học đã học:
- Mã môn: MATH 151
- Tên môn: Toán ứng dụng 1
- Số tín chỉ: 4
- Điểm: A-

Môn học được chuyển:
- Mã môn: MATH 101
- Tên môn: Giải tích 1A
- Số tín chỉ được chuyển: 4
- Điểm chuyển đổi: A
```

---

# 22. Generic table fallback

Không phải mọi bảng đều là bảng chuyển điểm.

Do đó cần generic serializer:

```text
<context>

Column A: value
Column B: value
Column C: value
...
```

Nếu table schema không map được domain-specific:

```python
serializer = GenericKeyValueTableSerializer()
```

Nếu nhận diện được bảng chuyển điểm:

```python
serializer = CourseTransferTableSerializer()
```

Thiết kế theo strategy pattern nếu thuận tiện.

---

# 23. Table row là atomic unit

Mỗi row ban đầu là một:

```text
AtomicUnit(type="table_row")
```

Sau đó Token-Aware Packing quyết định:

```text
1 row / chunk
```

hay:

```text
multiple very short rows / chunk
```

### Rule mặc định cho dữ liệu hiện tại

Với bảng chuyển đổi môn:

```text
prefer one semantic row = one child chunk
```

vì mỗi row thể hiện một mapping độc lập.

---

# 24. AtomicUnit model

Tạo intermediate model:

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

Ví dụ unit types:

```text
legal_basis
article_intro
clause
point
paragraph
table_row
recipient_item
signature
```

---

# 25. Token-Aware Packing

Không dùng:

```python
len(text)
```

để quyết định chunk size.

Bắt buộc dùng tokenizer thật của embedding model.

---

# 26. Token counter

Tạo:

```python
class TokenCounter:
    def __init__(self, model):
        ...

    def count(self, text: str) -> int:
        ...
```

Có thể lấy tokenizer từ SentenceTransformer underlying tokenizer.

Ví dụ:

```python
tokenizer = model.tokenizer
```

hoặc:

```python
tokenizer = model._first_module().tokenizer
```

nhưng phải code defensive vì SentenceTransformer versions khác nhau.

Ưu tiên official public attributes nếu có.

---

# 27. Xác định token budget

Không hard-code `256` nếu model có thể cung cấp.

Tạo:

```python
max_seq_length = model.max_seq_length
```

Config:

```python
TOKEN_SAFETY_MARGIN = 16
```

Tính:

```python
available_budget = (
    max_seq_length
    - contextual_prefix_tokens
    - special_token_margin
    - TOKEN_SAFETY_MARGIN
)
```

Không được để:

```text
retrieval_text token count > model.max_seq_length
```

---

# 28. Prefix-aware token budget

Do mỗi child sẽ có contextual prefix:

```text
Văn bản: ...
Số: ...
Cơ quan: ...
Phần: ...
```

nên phải count cả prefix.

Không được:

```text
body <= 256
+
prefix 50
=
306
```

rồi để model truncate.

---

# 29. Packing algorithm

Input:

```text
list[AtomicUnit]
```

Output:

```text
list[PackedUnit]
```

Pseudo:

```python
current = []

for unit in units:
    candidate = current + [unit]

    if token_count(build_candidate_text(candidate)) <= budget:
        current.append(unit)
        continue

    if current:
        emit(current)
        current = []

    if token_count(unit.text) <= budget:
        current = [unit]
    else:
        emit(recursive_split_atomic_unit(unit))

if current:
    emit(current)
```

---

# 30. Boundary priority khi unit quá dài

Nếu một AtomicUnit tự nó vượt budget, split theo thứ tự:

```text
Article
→ Clause
→ Point
→ Paragraph
→ Sentence
→ token window fallback
```

Không dùng token window trước khi thử semantic boundary.

---

# 31. Sentence fallback

Có thể dùng regex tiếng Việt đơn giản hoặc sentence tokenizer.

Không thêm dependency nặng nếu chưa cần.

Fallback cuối:

```python
split_by_token_window(...)
```

nhưng phải:

```text
preserve parent_id
preserve section_path
mark chunk metadata:
    "split_fallback": "token_window"
```

---

# 32. Overlap policy

Không dùng fixed overlap mặc định kiểu:

```text
20% token overlap
```

Ưu tiên contextual overlap thông qua:

```text
document metadata
section path
article title
table schema
```

Nếu cần overlap cho long prose fallback:

```python
FALLBACK_TOKEN_OVERLAP = 20
```

chỉ dùng ở `token_window fallback`, không dùng cho Điều/Table row bình thường.

---

# 33. Parent / Child model

## 33.1 Parent

Parent đại diện context rộng:

```text
Article
Table
Preamble section
```

Ví dụ:

```text
parent_article_1
```

Parent không bắt buộc có vector trong v2.

Lưu:

```text
parent_id
parent_type
text
page_start
page_end
section_path
child_ids
```

---

## 33.2 Child

Child là unit được embedding/search.

Ví dụ:

```text
child_article1_intro
child_article1_table_row_01
child_article1_table_row_02
child_article1_paragraph_01
```

Mỗi child có:

```text
parent_id
```

---

# 34. Quy tắc parent

### Article text

```text
Parent = Điều
```

### Table row nằm trong Điều

Khuyến nghị:

```text
Parent = Điều
table_id = logical table
```

Có thể có secondary logical parent:

```text
table_parent_id
```

Payload:

```json
{
  "parent_id": "article_1",
  "table_parent_id": "table_1"
}
```

### Legal basis

```text
Parent = preamble
```

---

# 35. Retrieval text

Không embed `raw_text` trực tiếp.

Tạo:

```python
def build_retrieval_text(
    document_meta: dict,
    chunk: Chunk,
) -> str:
    ...
```

Format mặc định:

```text
Văn bản: {Summary}
Số: {No}
Cơ quan ban hành: {Author}
Ngày: {DateDocument}
Phần: {section_path joined by " > "}

Nội dung:
{normalized_text}
```

---

# 36. Metadata null handling

Không được để:

```text
Văn bản: None
Số: None
```

Tạo helper:

```python
def safe_meta(value):
    return value.strip() if value else ""
```

Chỉ thêm field nếu có giá trị.

---

# 37. Retrieval text cho table row

Table row cần context rõ hơn:

```text
Văn bản: ...
Phần: Điều 1 > Bảng chuyển điểm
Đối tượng: Võ Hoàng Duy   # nếu derive chắc chắn từ Summary/article text

Nội dung:
...
```

Không được hallucinate tên đối tượng.

Chỉ lấy từ:

- `Summary`;
- metadata;
- text Article đã parse;
- table header.

---

# 38. Context không được quá dài

Không prepend toàn bộ parent vào từng child.

Ví dụ không làm:

```text
full Article 1
+
row
```

vì sẽ phá token budget và duplicate lớn.

Chỉ prepend:

```text
document-level metadata
+
section path
+
short structural label
```

---

# 39. Qdrant payload schema

Child point payload đề nghị:

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

  "chunk_id": "...",
  "chunk_index": 7,
  "chunk_type": "table_row",

  "parent_id": "article_1",
  "table_parent_id": "table_1",

  "section_path": [
    "QUYẾT ĐỊNH",
    "Điều 1",
    "Bảng chuyển điểm"
  ],

  "page_start": 1,
  "page_end": 2,

  "raw_text": "...",
  "normalized_text": "...",
  "retrieval_text": "...",

  "token_count": 143,

  "source_block_ids": [
    "page_001_block_012"
  ],

  "table_id": "logical_table_01",
  "table_row_index": 3,

  "source": "OCR_JSON",
  "chunking_version": "v2"
}
```

---

# 40. Parent storage

Có hai lựa chọn.

## Option A — Qdrant non-vector separate collection

Ví dụ:

```text
rag_document_parents
```

Nhưng Qdrant vẫn thiên về vector points.

## Option B — lưu parent trong payload/database riêng

Trong v2 có thể đơn giản:

- child vector trong Qdrant;
- parent text được lưu trong JSON/DB/cache;
- hoặc parent cũng là point trong Qdrant nhưng `is_parent=True`.

### Đề nghị cho implementation đầu tiên

Dùng cùng collection nếu dễ vận hành:

```json
{
  "record_type": "child"
}
```

Parent có thể lưu:

```json
{
  "record_type": "parent"
}
```

Nhưng chỉ child có vector retrieval active.

Nếu Qdrant client yêu cầu vector cho mọi point, cân nhắc:

- separate parent store;
- hoặc separate collection.

Không dùng zero-vector parent chung collection nếu có nguy cơ bị search.

---

# 41. Stable IDs

Code hiện tại dùng UUID5:

```python
uuid.uuid5(uuid.UUID(META["Id"]), f"chunk-{idx}")
```

Giữ tính deterministic.

Đề nghị:

```python
chunk_uuid = uuid.uuid5(
    uuid.UUID(document_id),
    f"v2:{chunk_type}:{section_path}:{chunk_index}:{source_signature}"
)
```

Parent:

```python
parent_uuid = uuid.uuid5(
    uuid.UUID(document_id),
    f"v2:parent:{node_type}:{node_index}"
)
```

Không chỉ dựa vào chunk index nếu có thể, vì index có thể thay đổi khi parser thay đổi nhẹ.

---

# 42. Integrate vào embedding pipeline

Refactor `load_data()`.

Pseudo:

```python
def load_data(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            meta = build_meta(row)

            json_path = resolve_ocr_json_path(meta)

            chunks = build_document_chunks(
                json_path=json_path,
                document_meta=meta,
                embedding_model=model,
            )

            child_chunks = [
                c for c in chunks
                if c.metadata.get("is_embedding_child", True)
            ]

            points = []

            for chunk in child_chunks:
                assert chunk.token_count <= model.max_seq_length

                vector = model.encode(
                    chunk.retrieval_text,
                    normalize_embeddings=True
                )

                points.append(
                    PointStruct(
                        id=chunk.chunk_id,
                        vector=vector.tolist(),
                        payload=chunk_to_payload(chunk, meta)
                    )
                )

            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
```

---

# 43. JSON path resolution

Hiện tại:

```text
./data/file_contents/{Id}.txt
```

Cần cấu hình tương đương:

```text
./data/file_contents/{Id}.json
```

hoặc location thật trong repo.

Không hard-code nhiều nơi.

Config:

```python
OCR_JSON_DIR = Path("./data/file_contents")
```

Helper:

```python
def resolve_ocr_json_path(meta: dict) -> Path:
    return OCR_JSON_DIR / f"{meta['Id']}.json"
```

Nếu JSON không tồn tại:

```text
log warning
→ optional fallback V1 TXT
```

Fallback phải explicit:

```python
ENABLE_V1_TXT_FALLBACK = True
```

---

# 44. Retrieval API v2

Current:

```python
def embedding_search(query: str, top_k: int=5):
```

Tạo mới:

```python
def embedding_search_v2(
    query: str,
    top_k: int = 10,
    expand_context: bool = True,
):
    ...
```

Flow:

```text
embed query
↓
search CHILD chunks
↓
filter record_type=child
↓
deduplicate
↓
optional rerank
↓
expand parent/siblings
↓
return retrieval result objects
```

---

# 45. Deduplication

Do một document có nhiều child gần giống, cần tránh top-k toàn cùng một parent.

Có thể hỗ trợ:

```python
max_children_per_parent = 3
```

Hoặc:

```text
group by parent_id
```

Không bắt buộc luôn bật; cấu hình được.

---

# 46. Parent expansion

Tạo:

```python
def expand_context(
    retrieved_children,
    parent_store,
    strategy="adaptive",
):
    ...
```

Các strategy:

```text
none
parent
siblings
adaptive
```

### v2 default

```text
adaptive
```

Heuristic đơn giản:

- table row query chính xác:
  - matched child;
  - table header/schema;
  - không cần full parent nếu đủ context.

- general Article query:
  - expand parent Article.

- nhiều child cùng parent được hit:
  - có thể merge/expand parent.

---

# 47. Không cần LLM router ở v2

Không thêm LLM vào chunking pipeline v2 chỉ để quyết định boundary.

Ưu tiên deterministic parser:

```text
layout
regex
document structure
table schema
tokenizer
```

LLM-based semantic chunking có thể benchmark ở phase sau.

---

# 48. Logging

Mỗi document log:

```text
document_id
page_count
block_count
noise_blocks
article_count
legal_basis_count
table_count
cross_page_table_count
parent_count
child_count
max_child_tokens
avg_child_tokens
fallback_split_count
```

Ví dụ:

```text
[chunking-v2]
doc=...
articles=3
tables=1
cross_page_tables=1
children=17
max_tokens=211
fallback_token_splits=0
```

---

# 49. Validation bắt buộc

Tạo:

```python
def validate_chunks(chunks, model):
    ...
```

Assert/raise nếu:

### 49.1 Token overflow

```python
token_count(retrieval_text) > model.max_seq_length
```

=> ERROR.

Không cho phép silent truncation.

### 49.2 Missing parent

Nếu child có:

```text
parent_id
```

thì parent phải tồn tại.

### 49.3 Table row context

Nếu:

```text
chunk_type == "table_row"
```

thì phải có:

```text
table_id
section_path
```

### 49.4 Empty retrieval text

Không upsert empty/near-empty chunks.

---

# 50. Unit tests — Structure parser

Dùng các OCR JSON mẫu hiện có.

## Test A — Quyết định Hồ Xuân Tường

Expected:

```text
detect Decision
detect Điều 1
detect Điều 2
detect Điều 3
detect Điều 4
detect Nơi nhận
```

Phần căn cứ phải nằm trước Điều 1.

Không để:

```text
Nơi nhận
```

thuộc Điều 4.

---

# 51. Unit tests — Cross-page table

## Test B — Phạm Minh Quân

Expected:

```text
Article 1
  └── logical table
      page_start = 1
      page_end = 2
```

Rows ở page 2 như:

```text
PHYS 201
MATH 151
CSE 101
...
```

phải dùng schema của table bắt đầu ở page 1.

Không tạo một table semantic độc lập mất header.

---

# 52. Unit tests — Cross-page table Võ Hoàng Duy

Expected:

```text
Article 1
  └── table page 1 + table page 2
```

Row:

```text
PHYS 201
Vật lý 1A
...
```

page 2 vẫn phải có:

```text
source course
target course
```

semantic column mapping.

---

# 53. Unit tests — Table semantic row

Input row:

```text
MATH 151
Toán ứng dụng 1
4
A.
MATH 101
Giải tích 1A
4
A
```

Expected retrieval text chứa ít nhất:

```text
MATH 151
Toán ứng dụng 1
MATH 101
Giải tích 1A
```

và labels phân biệt source/target.

Không chỉ:

```text
MATH 151 | Toán ứng dụng 1 | ...
```

---

# 54. Unit tests — Token budget

Với mọi child:

```python
assert tokenizer_count(chunk.retrieval_text) <= model.max_seq_length
```

Test phải sử dụng tokenizer thật của embedding model nếu model có sẵn trong test environment.

Nếu test CI không muốn tải model, cho phép dependency injection fake tokenizer cho unit test, nhưng phải có ít nhất một integration test bằng tokenizer thật.

---

# 55. Unit tests — No accidental Article split

Text:

```text
Điều 2: Sinh viên có tên ở điều 1 phải đóng...
```

Expected:

```text
1 Article boundary at "Điều 2"
```

Không tạo Article 1 mới từ substring:

```text
ở điều 1 phải
```

---

# 56. End-to-end debug output

Tạo optional CLI:

```bash
python -m chunking.debug \
  --input path/to/file.json \
  --meta path/to/meta.json \
  --output chunks_debug.json
```

Output:

```json
{
  "document": {...},
  "parents": [...],
  "children": [...]
}
```

Mục đích:

- inspect chunking trước khi embedding;
- benchmark;
- phát hiện parser lỗi.

---

# 57. Debug Markdown output

Khuyến nghị thêm:

```bash
--output-markdown chunks_debug.md
```

Ví dụ:

```markdown
# Document

## Parent: Điều 1

### Child 1 — article_intro
...

### Child 2 — table_row
...

### Child 3 — table_row
...
```

Rất hữu ích để review thủ công.

---

# 58. Configuration

Tạo config object:

```python
@dataclass
class ChunkingConfig:
    safety_margin_tokens: int = 16
    fallback_overlap_tokens: int = 20

    index_recipients: bool = True
    index_signature: bool = True

    merge_cross_page_tables: bool = True

    prefer_single_table_row_chunks: bool = True

    enable_v1_txt_fallback: bool = True
```

Không rải magic numbers trong code.

---

# 59. Backward compatibility

Giữ:

```python
chunk_legal_document()
```

tạm thời để benchmark baseline.

Đổi tên rõ:

```python
chunk_legal_document_v1()
```

Tạo:

```python
build_document_chunks_v2()
```

Không xóa baseline ngay.

---

# 60. Feature flag

Cho phép:

```python
CHUNKING_VERSION = "v2"
```

hoặc env:

```bash
CHUNKING_VERSION=v2
```

Flow:

```python
if CHUNKING_VERSION == "v1":
    ...
elif CHUNKING_VERSION == "v2":
    ...
```

Mục tiêu:

- A/B test;
- rollback;
- benchmark.

---

# 61. Chunking version trong payload

Mọi point phải có:

```json
{
  "chunking_version": "v2"
}
```

Không trộn khó phân biệt với index cũ.

Khuyến nghị collection mới:

```text
rag_document_v2
```

thay vì overwrite production collection cũ ngay.

---

# 62. Không destroy collection hiện tại

Không chạy:

```python
recreate_collection(...)
```

hoặc delete current production collection tự động.

Tạo config:

```text
COLLECTION_NAME_V2="rag_document_v2"
```

Migration phải explicit.

---

# 63. Evaluation dataset

Sau implementation, tạo tập queries nhỏ tối thiểu cho 3 documents mẫu.

Các loại query:

## Metadata

```text
Quyết định chuyển điểm cho Võ Hoàng Duy là gì?
```

## Article

```text
Điều 2 của quyết định Võ Hoàng Duy quy định gì?
```

## Table lookup

```text
Võ Hoàng Duy môn Toán ứng dụng 1 được chuyển thành môn gì?
```

## Grade

```text
Điểm chuyển đổi môn MATH 151 của Võ Hoàng Duy là bao nhiêu?
```

## Cross-page table

```text
Võ Hoàng Duy môn PHYS 201 được chuyển thế nào?
```

## Entity distinction

```text
MATH 151 của Phạm Minh Quân được chuyển sang môn nào?
```

và:

```text
MATH 151 của Võ Hoàng Duy được chuyển sang môn nào?
```

Retriever phải ưu tiên đúng document theo tên sinh viên.

---

# 64. Evaluation metrics

Tối thiểu:

```text
Recall@1
Recall@3
Recall@5
MRR
```

Nếu có relevance grading:

```text
nDCG@5
```

Thêm engineering metrics:

```text
avg chunk tokens
p95 chunk tokens
max chunk tokens
child count/document
index size
retrieval latency
```

---

# 65. Baseline comparison

Benchmark cùng:

```text
embedding model
query set
Qdrant distance
top_k
```

Chỉ thay chunker.

Compare:

```text
V1 = current Điều regex
V2 = Structure/Table/ParentChild/TokenAware
```

Không đổi embedding model trong cùng benchmark đầu tiên vì sẽ không biết gain đến từ đâu.

---

# 66. Acceptance criteria

Implementation chỉ được coi là hoàn tất khi thỏa tất cả:

## Structure

- [ ] Parse được Điều 1/2/3/4 trong document mẫu phù hợp.
- [ ] Legal basis không bị gom vô hạn vào một mega chunk.
- [ ] Table thuộc đúng Điều.
- [ ] Nơi nhận không thuộc Điều cuối.

## Table

- [ ] Parse HTML table.
- [ ] Multi-row header có representation semantic.
- [ ] Detect được ít nhất các cross-page tables trong documents mẫu.
- [ ] Row page 2 kế thừa schema page 1.
- [ ] Không embed raw HTML table làm chunk chính.

## Token

- [ ] Count bằng tokenizer thật.
- [ ] Không child nào vượt max model tokens.
- [ ] Không còn logic core `len(text) > 3000`.

## Parent/Child

- [ ] Mỗi child có parent_id hợp lệ.
- [ ] Table row biết Điều cha.
- [ ] Retrieval có thể expand parent/siblings.

## Storage

- [ ] Payload có `section_path`.
- [ ] Payload có `chunk_type`.
- [ ] Payload có `page_start/page_end`.
- [ ] Payload có `raw_text`.
- [ ] Payload có `retrieval_text`.
- [ ] Payload có `token_count`.
- [ ] Payload có `chunking_version=v2`.

## Safety

- [ ] Không tự overwrite collection production.
- [ ] Không silent truncation.
- [ ] Không tự sửa raw OCR.
- [ ] Không hallucinate missing table values.

---

# 67. Non-goals của v2

Không triển khai trong scope đầu tiên:

```text
LLM semantic chunk boundary detection
Late Chunking
Mixture-of-Chunkers
query-dependent chunking
multimodal image embeddings
OCR correction bằng LLM
graph RAG
```

Những phần này là experiment phase sau.

V2 phải ưu tiên:

```text
deterministic
auditable
reproducible
layout-aware
table-aware
token-safe
```

---

# 68. Implementation priority

Codex nên thực hiện theo đúng thứ tự:

## Phase 1 — Foundation

1. `models.py`
2. `normalize.py`
3. `ocr_parser.py`
4. tokenizer/token counter
5. baseline tests

## Phase 2 — Structure

6. `structure_parser.py`
7. detect Điều
8. detect legal basis
9. detect recipients/signature
10. build document tree

## Phase 3 — Tables

11. parse HTML
12. flatten schema
13. semantic row serializer
14. cross-page table detection
15. cross-page merge tests

## Phase 4 — Chunk creation

16. AtomicUnit
17. Token-Aware Packing
18. Parent creation
19. Child creation
20. retrieval text

## Phase 5 — Integration

21. integrate `embbeding.py`
22. create Qdrant v2 payload
23. new collection/config
24. retrieval v2
25. parent expansion

## Phase 6 — Evaluation

26. debug JSON/Markdown
27. run 3 sample documents
28. inspect chunks
29. run baseline vs v2 retrieval test

---

# 69. Coding rules cho Codex

1. Viết code type-hinted.
2. Tách pure functions tối đa có thể.
3. Không hard-code document IDs mẫu trong production code.
4. Không dùng LLM API trong chunking v2.
5. Không thêm dependency lớn nếu standard library/bs4 đủ.
6. Mọi heuristic phải:
   - nằm trong function có tên rõ;
   - có comment;
   - có test.
7. Không sửa raw OCR.
8. Không silent fallback khi parser lỗi nghiêm trọng.
9. Log warning khi:
   - table parse fail;
   - cross-page match uncertain;
   - token fallback split được dùng.
10. Ưu tiên deterministic output.

---

# 70. Definition of Done

Khi hoàn tất, Codex phải cung cấp:

```text
1. Danh sách file đã tạo/sửa
2. Mô tả ngắn kiến trúc implementation
3. Test command
4. Kết quả test
5. Ví dụ chunks của 3 OCR JSON mẫu
6. Thống kê token
7. Ví dụ table row serialization
8. Ví dụ cross-page table merge
9. Ví dụ Qdrant payload
10. Các hạn chế còn lại
```

Không chỉ trả về:

```text
"Implemented successfully"
```

mà không có evidence.

---

# 71. Prompt hành động cuối cùng cho Codex

Thực hiện implementation dựa trên toàn bộ đặc tả này.

Trước khi sửa code:

1. Đọc file `embbeding.py` hoặc `embedding_v1.py` hiện có.
2. Xác định structure repository.
3. Xác định đường dẫn thực tế của OCR JSON.
4. Không phá API hiện tại nếu không cần.
5. Giữ baseline chunker v1 để comparison.

Sau đó triển khai theo từng phase.

Nếu một yêu cầu trong tài liệu này xung đột với implementation hiện tại:

- ưu tiên backward compatibility khi hợp lý;
- nhưng **không được bỏ** bốn nguyên lý cốt lõi:

```text
Structure-Aware
Table-Aware
Parent/Child
Token-Aware
```

Nếu cần đơn giản hóa một chi tiết kỹ thuật, phải ghi rõ:

```text
what changed
why
impact
follow-up
```

---

# 72. Expected architecture sau khi hoàn tất

```text
documents.csv
     │
     ▼
document metadata
     │
     ├──────────────┐
     │              │
     ▼              ▼
OCR JSON       metadata context
     │
     ▼
OCR parser
     │
     ▼
normalized blocks
     │
     ▼
structure parser
     │
     ▼
document tree
     │
     ├───────────────┐
     │               │
     ▼               ▼
text units       table parser
                     │
                     ▼
              cross-page tables
                     │
                     ▼
                table rows
     │               │
     └───────┬───────┘
             ▼
         atomic units
             │
             ▼
       token-aware packer
             │
             ▼
       parent/child builder
             │
             ▼
       retrieval text builder
             │
             ▼
     tokenizer validation
             │
             ▼
          embedding
             │
             ▼
       Qdrant V2 collection
             │
             ▼
          child search
             │
             ▼
    parent/sibling expansion
             │
             ▼
           RAG LLM
```

---

# 73. Nguyên tắc quan trọng nhất

Nếu phải nhớ một điều duy nhất:

> **Không chunk tài liệu hành chính như một chuỗi ký tự. Hãy parse cấu trúc tài liệu trước, tạo semantic atomic units, sau đó mới dùng token budget để đóng gói các units thành chunk.**

Với table:

> **Không embed một bảng lớn dưới dạng raw HTML nếu mục tiêu retrieval nằm ở từng row. Hãy bảo toàn schema + row relationship và context Điều/document.**

Với retrieval:

> **Dùng child nhỏ để tìm chính xác, nhưng dùng parent/siblings để cấp đủ context cho LLM.**

Với tokenizer:

> **Không có chunk nào được phép vượt context của embedding model và bị truncate âm thầm.**

---

# 74. Tên đề xuất cho version

Internal name:

```text
SAHC-v2
```

Meaning:

```text
Structure-Aware Hierarchical Chunking v2
```

Full architecture:

```text
SAHC-v2 =
Structure-Aware Hierarchical Chunking
+ Table-Aware Row Chunking
+ Parent/Child Retrieval
+ Token-Aware Packing
```

Đây là version nên dùng làm candidate chính để benchmark với baseline `chunk_legal_document_v1`.
