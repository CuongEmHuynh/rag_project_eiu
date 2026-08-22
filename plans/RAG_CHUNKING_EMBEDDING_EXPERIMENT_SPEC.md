# RAG Chunking × Embedding Experimental Framework

**Version:** 1.0  
**Purpose:** Thiết kế framework thực nghiệm có thể switch giữa tối đa 4 phương pháp chunking và nhiều embedding model, index vào Qdrant, sau đó đánh giá để tìm tổ hợp **Chunking Method × Embedding Model × Chunk Length** tốt nhất cho dữ liệu OCR tiếng Việt.

---

# 1. Mục tiêu

Xây dựng một framework thực nghiệm production-oriented nhưng vẫn đủ chặt chẽ cho nghiên cứu, với các yêu cầu:

1. Từ tập dữ liệu OCR hiện tại, hỗ trợ switch qua lại tối đa 4 phương pháp chunking.
2. Sau chunking, hỗ trợ switch nhiều embedding model mà không thay đổi chunking pipeline.
3. Mỗi tổ hợp chunking × embedding có thể index độc lập vào Qdrant.
4. Có compatibility guard để ngăn chunk vượt quá context length của embedding model.
5. Có thể benchmark theo nhiều chunk length.
6. Đánh giá bằng các retrieval metrics phổ biến như Recall@K, HitRate@K, MRR, nDCG@K, MAP.
7. Theo dõi thêm efficiency metrics như số lượng chunk, kích thước chunk, thời gian embedding, throughput, dung lượng index, query latency.
8. Không để xảy ra silent truncation trong benchmark.
9. Có khả năng mở rộng để đưa SAHC, Late Chunking, Hybrid Search và Reranker vào các phase sau.

---

# 2. Triết lý thiết kế thí nghiệm

Không thiết kế pipeline kiểu:

```text
Fixed -> BGE
Semantic -> Jina
Hierarchical -> Qwen
```

Vì nếu kết quả thay đổi sẽ không xác định được nguyên nhân đến từ chunking hay embedding.

Thiết kế đúng là ma trận thực nghiệm:

```text
                      Embedding
                +--------+--------+--------+--------+
                | BGE-M3 | GTE    | Qwen3  | Jina   |
+---------------+--------+--------+--------+--------+
| Fixed         |   X    |   X    |   X    |   X    |
| Recursive     |   X    |   X    |   X    |   X    |
| Semantic      |   X    |   X    |   X    |   X    |
| SAHC/Hierarch |   X    |   X    |   X    |   X    |
+---------------+--------+--------+--------+--------+
```

Nếu dùng 4 chunker và 4 embedding model thì có tối thiểu 16 tổ hợp. Nếu benchmark thêm nhiều chunk length thì không gian thí nghiệm là:

```text
Chunker × Embedding Model × Chunk Length
```

Ví dụ:

```text
4 × 4 × 4 = 64 experiments
```

---

# 3. Phân loại đúng các kỹ thuật

## 3.1 Boundary strategies

Các phương pháp quyết định vị trí cắt:

- Fixed-size Token Chunking
- Recursive Splitting
- Semantic Chunking
- SAHC hoặc Hierarchical/Structure-aware Chunking

## 3.2 Embedding strategy

Late Chunking không nên được coi như một splitter ngang hàng với bốn phương pháp trên.

Nó nên được biểu diễn dưới dạng:

```yaml
embedding_mode: normal
```

hoặc:

```yaml
embedding_mode: late_chunking
```

Late Chunking thay đổi cách tạo vector embedding: Transformer đọc context dài trước, sau đó mới pooling token theo chunk boundary.

## 3.3 Retrieval topology

Hierarchical retrieval/parent-child retrieval cũng nên được tách khỏi boundary detection khi làm ablation:

```text
Flat retrieval
vs
Parent-child retrieval
vs
Auto-merging retrieval
```

---

# 4. Bốn phương pháp chunking cho vòng thực nghiệm đầu tiên

Khuyến nghị:

```text
1. Fixed Token Chunking
2. Recursive Splitting
3. Semantic Chunking
4. SAHC hoặc Hierarchical/Structure-aware Chunking
```

Nếu mục tiêu nghiên cứu là chứng minh đóng góp SAHC, nên dùng:

```text
Fixed
Recursive
Semantic
SAHC
```

Thay vì dùng một hierarchical generic baseline ở slot thứ tư.

---

# 5. Nguyên tắc về chunk length

Không dùng nguyên tắc:

> Chunk càng dài càng tốt.

Chunk quá dài có thể gây semantic dilution: một vector chứa quá nhiều chủ đề không liên quan.

Mục tiêu thực tế:

```text
Maximize retrieval quality subject to model context constraints.
```

Hay:

```text
BestLength = argmax RetrievalScore(length)
```

với ràng buộc:

```text
chunk_tokens <= embedding_model_max_tokens
```

Không được lấy `max_context_length` của model làm chunk size mặc định.

Nên benchmark theo nhiều mức, ví dụ:

```yaml
chunk_lengths:
  - 256
  - 512
  - 1024
  - 2048
  - 4096
```

Với model long-context có thể thử thêm:

```yaml
  - 8192
  - 16384
  - 32768
```

Nhưng chỉ với model tương thích.

---

# 6. Hai track thí nghiệm bắt buộc

## 6.1 Track A — Fair Chunking Benchmark

Mục tiêu: so sánh chunking method công bằng.

Giữ cố định:

- Dataset
- OCR normalization
- Embedding model
- Vector DB
- Distance metric
- Retrieval top-k
- Reranker, nếu có
- Query set
- Metric implementation

Thay đổi duy nhất:

```text
Chunking Method
```

Ví dụ:

```text
Fixed-512
Recursive-512
Semantic với chunk-size control tương đương
SAHC với token-budget control tương đương
```

Lưu ý: Semantic và SAHC là dynamic chunking nên không thể ép mọi chunk bằng 512 tokens, nhưng cần đặt `min_tokens`, `target_tokens`, `max_tokens` sao cho phân phối length có thể so sánh.

## 6.2 Track B — Natural / Maximum Context Benchmark

Cho phép mỗi phương pháp hoạt động theo cấu hình tự nhiên và khai thác context length của embedding model.

Mục tiêu:

```text
Tìm tổ hợp thực tế tốt nhất cho production.
```

Ví dụ:

```text
SAHC + BGE-M3 + max_tokens=4096
SAHC + Qwen3 + max_tokens=8192
Semantic + Qwen3 + max_tokens=4096
...
```

Track này không dùng để chứng minh riêng contribution của chunking boundary, mà dùng để tìm hệ thống tối ưu tổng thể.

---

# 7. Embedding models đề xuất

Framework không được hardcode model. Mỗi model phải có metadata riêng.

Candidate ban đầu:

```text
BAAI/bge-m3
Alibaba-NLP/gte-multilingual-base
Qwen/Qwen3-Embedding-0.6B
jinaai/jina-embeddings-v4
```

Có thể thêm baseline:

```text
intfloat/multilingual-e5-large-instruct
```

Nhưng model E5 có context ngắn hơn nên phù hợp controlled 512-token benchmark hơn là long-context benchmark.

Metadata cần lưu cho từng model:

```text
name
model_name
dimension
max_tokens
normalize
query_instruction
passage_instruction
trust_remote_code
supports_late_chunking
```

---

# 8. Kiến trúc tổng thể

```text
                         OCR Dataset
                             |
                             v
                    Document Loader
                             |
                             v
                    OCR Normalizer
                             |
                             v
                  +-------------------+
                  | Chunking Registry |
                  +---------+---------+
                            |
       +--------------------+---------------------+
       |                    |                     |
       v                    v                     v
     Fixed              Recursive             Semantic
                                                    |
                                          SAHC -----+
                            |
                            v
                       Chunk Dataset
                            |
                            v
                 Chunk Length Validator
                            |
                            v
                 +-------------------+
                 | Embedding Registry|
                 +---------+---------+
                           |
      +--------------------+----------------------+
      |                    |                      |
      v                    v                      v
   BGE-M3              GTE-multi              Qwen3
                                                 |
                                       Jina v4 --+
                           |
                           v
                      Embeddings
                           |
                           v
                 Qdrant Collection
                           |
                           v
                    Retrieval Test
                           |
                           v
       Recall@K / MRR / nDCG / Hit@K / MAP
                           |
                           v
                    Experiment Table
```

---

# 9. Project structure

Agent phải tạo project structure sau:

```text
rag_chunking_benchmark/
|
+-- configs/
|   +-- chunkers.yaml
|   +-- embedders.yaml
|   +-- experiments.yaml
|   +-- qdrant.yaml
|
+-- data/
|   +-- raw/
|   +-- normalized/
|   +-- chunks/
|   +-- gold/
|
+-- artifacts/
|   +-- chunks/
|   +-- embeddings/
|   +-- metrics/
|   +-- reports/
|
+-- src/
|   |
|   +-- documents/
|   |   +-- __init__.py
|   |   +-- loader.py
|   |   +-- normalizer.py
|   |   +-- schema.py
|   |
|   +-- chunkers/
|   |   +-- __init__.py
|   |   +-- base.py
|   |   +-- registry.py
|   |   +-- fixed.py
|   |   +-- recursive.py
|   |   +-- semantic.py
|   |   +-- sahc.py
|   |
|   +-- embeddings/
|   |   +-- __init__.py
|   |   +-- base.py
|   |   +-- registry.py
|   |   +-- specs.py
|   |   +-- huggingface.py
|   |   +-- validator.py
|   |
|   +-- vectorstores/
|   |   +-- __init__.py
|   |   +-- qdrant_store.py
|   |
|   +-- evaluation/
|   |   +-- __init__.py
|   |   +-- dataset.py
|   |   +-- retrieval.py
|   |   +-- metrics.py
|   |   +-- report.py
|   |
|   +-- pipeline/
|       +-- __init__.py
|       +-- chunking_pipeline.py
|       +-- embedding_pipeline.py
|       +-- indexing_pipeline.py
|       +-- retrieval_pipeline.py
|       +-- experiment.py
|
+-- tests/
|   +-- test_fixed_chunker.py
|   +-- test_recursive_chunker.py
|   +-- test_semantic_chunker.py
|   +-- test_sahc_chunker.py
|   +-- test_embedding_validator.py
|   +-- test_metrics.py
|
+-- scripts/
|   +-- run_chunking.py
|   +-- run_embedding.py
|   +-- run_indexing.py
|   +-- run_evaluation.py
|   +-- run_experiments.py
|
+-- requirements.txt
+-- pyproject.toml
+-- README.md
+-- main.py
```

---

# 10. Common document schema

Tạo `src/documents/schema.py`.

```python
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Document:
    document_id: str
    text: str
    source_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    chunk_method: str

    start_char: int = -1
    end_char: int = -1

    page_start: Optional[int] = None
    page_end: Optional[int] = None

    parent_id: Optional[str] = None
    level: Optional[int] = None

    metadata: dict[str, Any] = field(default_factory=dict)
```

Yêu cầu:

- Mọi chunker phải trả về `list[Chunk]`.
- Không tạo schema riêng cho từng chunker.
- Các metadata đặc thù được lưu vào `metadata`.

---

# 11. Base Chunker Interface

Tạo `src/chunkers/base.py`.

```python
from abc import ABC, abstractmethod
from src.documents.schema import Document, Chunk


class BaseChunker(ABC):
    name: str

    @abstractmethod
    def split(self, document: Document) -> list[Chunk]:
        raise NotImplementedError
```

Acceptance criteria:

- Tất cả chunker kế thừa `BaseChunker`.
- `split()` không được embedding hoặc ghi Qdrant.
- Chunking layer phải độc lập với vector DB.

---

# 12. Fixed Token Chunker

Tạo `src/chunkers/fixed.py`.

Yêu cầu:

- Dùng token count, không dùng character count làm metric chính.
- Configurable `chunk_size` và `overlap`.
- Không silent truncate.
- Lưu token length vào metadata.

Reference implementation:

```python
from src.chunkers.base import BaseChunker
from src.documents.schema import Document, Chunk


class FixedTokenChunker(BaseChunker):
    name = "fixed"

    def __init__(self, tokenizer, chunk_size: int = 512, overlap: int = 64):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: Document) -> list[Chunk]:
        token_ids = self.tokenizer.encode(
            document.text,
            add_special_tokens=False,
        )

        stride = self.chunk_size - self.overlap
        result: list[Chunk] = []

        for idx, start in enumerate(range(0, len(token_ids), stride)):
            end = min(start + self.chunk_size, len(token_ids))
            ids = token_ids[start:end]

            text = self.tokenizer.decode(
                ids,
                skip_special_tokens=True,
            )

            result.append(
                Chunk(
                    chunk_id=f"{document.document_id}__fixed__{idx:06d}",
                    document_id=document.document_id,
                    text=text,
                    chunk_method=self.name,
                    metadata={
                        "token_length": len(ids),
                        "chunk_size": self.chunk_size,
                        "overlap": self.overlap,
                    },
                )
            )

            if end >= len(token_ids):
                break

        return result
```

Tests:

- Empty input.
- Input shorter than chunk size.
- Input exactly equal chunk size.
- Input longer than chunk size.
- Overlap correctness.
- `overlap >= chunk_size` phải raise error.

---

# 13. Recursive Chunker

Tạo `src/chunkers/recursive.py`.

Có thể dùng `RecursiveCharacterTextSplitter`, nhưng length function phải đo bằng tokenizer.

Yêu cầu:

- Generic baseline.
- Không inject quá nhiều domain-specific separator trong fair benchmark.
- Cho phép cấu hình separators.

Example:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chunkers.base import BaseChunker
from src.documents.schema import Document, Chunk


class RecursiveTokenChunker(BaseChunker):
    name = "recursive"

    def __init__(
        self,
        tokenizer,
        chunk_size: int = 512,
        overlap: int = 64,
        separators: list[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=self.separators,
            length_function=self._token_length,
        )

    def _token_length(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def split(self, document: Document) -> list[Chunk]:
        texts = self.splitter.split_text(document.text)
        result = []

        for idx, text in enumerate(texts):
            result.append(
                Chunk(
                    chunk_id=f"{document.document_id}__recursive__{idx:06d}",
                    document_id=document.document_id,
                    text=text,
                    chunk_method=self.name,
                    metadata={
                        "token_length": self._token_length(text),
                        "chunk_size": self.chunk_size,
                        "overlap": self.overlap,
                    },
                )
            )

        return result
```

---

# 14. Semantic Chunker

Tạo `src/chunkers/semantic.py`.

Semantic Chunking phải tách hai khái niệm:

1. Atomic unit generation.
2. Semantic boundary detection.

Pipeline:

```text
Normalized OCR
    |
    v
Atomic Units
    |
    v
Context Window Construction
    |
    v
Unit Embeddings
    |
    v
Cosine Distance
    |
    v
Breakpoint Threshold
    |
    v
Chunk Grouping
```

Core idea:

```text
distance_i = 1 - cosine(embedding_i, embedding_{i+1})
```

Nếu distance vượt threshold thì tạo boundary.

Config:

```yaml
semantic:
  semantic_model: BAAI/bge-m3
  breakpoint_mode: percentile
  breakpoint_percentile: 90
  buffer_size: 1
  min_tokens: 256
  target_tokens: 512
  max_tokens: 1024
```

Yêu cầu quan trọng:

- Nếu semantic chunk vượt `max_tokens`, phải fallback split tiếp.
- Nếu chunk quá nhỏ, có thể merge với neighbor gần nhất nếu không phá `max_tokens`.
- Lưu distance/threshold metadata để debug.
- Semantic chunker model phục vụ boundary detection không nhất thiết phải giống retrieval embedding model.
- Nhưng trong benchmark phải freeze semantic-boundary model để tránh thêm một biến độc lập.

Pseudo-code:

```python
def semantic_chunking(text):
    units = split_atomic_units(text)
    windows = build_context_windows(units)
    embeddings = semantic_model.encode(windows, normalize_embeddings=True)

    similarities = cosine_adjacent(embeddings)
    distances = 1.0 - similarities
    threshold = percentile(distances, configured_percentile)

    boundaries = where(distances > threshold)
    chunks = group_units(units, boundaries)
    chunks = enforce_min_max_tokens(chunks)

    return chunks
```

Tests:

- Single sentence.
- Multiple semantically similar sentences.
- Strong topic shift.
- No breakpoint found.
- Very long semantic segment must split.
- Very short chunk merge behavior.

---

# 15. SAHC / Hierarchical Chunker

Tạo `src/chunkers/sahc.py`.

Nếu SAHC hiện tại đã tồn tại trong project khác, không rewrite toàn bộ ngay. Thay vào đó tạo adapter đưa output về Common Chunk schema.

Example:

```python
from src.chunkers.base import BaseChunker
from src.documents.schema import Document, Chunk


class SAHCChunker(BaseChunker):
    name = "sahc"

    def __init__(self, sahc_parser, tokenizer, max_tokens: int = 2048):
        self.sahc_parser = sahc_parser
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def split(self, document: Document) -> list[Chunk]:
        nodes = self.sahc_parser.parse(document.text)
        chunks = []

        for idx, node in enumerate(nodes):
            text = node.text
            token_length = len(self.tokenizer.encode(text, add_special_tokens=False))

            # nếu SAHC node quá dài phải split bằng safe fallback
            # nhưng vẫn giữ parent/section metadata

            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}__sahc__{idx:06d}",
                    document_id=document.document_id,
                    text=text,
                    chunk_method=self.name,
                    parent_id=getattr(node, "parent_id", None),
                    level=getattr(node, "level", None),
                    metadata={
                        "token_length": token_length,
                        "section_path": getattr(node, "section_path", None),
                    },
                )
            )

        return chunks
```

Yêu cầu:

- Preserve structural metadata.
- Có `section_path` nếu có.
- Có `parent_id` nếu có.
- Không loại bỏ hierarchy khi flatten thành chunks.
- Nếu một node vượt max token, split thành children nhưng preserve metadata path.

---

# 16. Chunker Registry

Tạo `src/chunkers/registry.py`.

```python
from src.chunkers.fixed import FixedTokenChunker
from src.chunkers.recursive import RecursiveTokenChunker
from src.chunkers.semantic import SemanticChunker
from src.chunkers.sahc import SAHCChunker


CHUNKER_REGISTRY = {
    "fixed": FixedTokenChunker,
    "recursive": RecursiveTokenChunker,
    "semantic": SemanticChunker,
    "sahc": SAHCChunker,
}


def create_chunker(method: str, **kwargs):
    if method not in CHUNKER_REGISTRY:
        supported = ", ".join(sorted(CHUNKER_REGISTRY))
        raise ValueError(f"Unsupported chunker={method}. Supported: {supported}")

    return CHUNKER_REGISTRY[method](**kwargs)
```

Acceptance criteria:

- Main pipeline không import trực tiếp từng chunker.
- Switch chunker chỉ bằng config.

---

# 17. Chunker configuration

Tạo `configs/chunkers.yaml`.

```yaml
active: recursive

fixed:
  chunk_size: 512
  overlap: 64

recursive:
  chunk_size: 512
  overlap: 64
  separators:
    - "\n\n"
    - "\n"
    - ". "
    - "; "
    - ", "
    - " "
    - ""

semantic:
  semantic_model: BAAI/bge-m3
  breakpoint_mode: percentile
  breakpoint_percentile: 90
  buffer_size: 1
  min_tokens: 256
  target_tokens: 512
  max_tokens: 1024

sahc:
  min_tokens: 128
  target_tokens: 512
  max_tokens: 2048
  fallback_splitter: recursive
```

---

# 18. Embedding model spec

Tạo `src/embeddings/specs.py`.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingSpec:
    name: str
    model_name: str
    dimension: int
    max_tokens: int

    normalize: bool = True
    trust_remote_code: bool = False

    query_instruction: str | None = None
    passage_instruction: str | None = None

    supports_late_chunking: bool = False
```

Registry ban đầu:

```python
EMBEDDING_SPECS = {
    "bge_m3": EmbeddingSpec(
        name="bge_m3",
        model_name="BAAI/bge-m3",
        dimension=1024,
        max_tokens=8192,
        normalize=True,
    ),

    "gte_multi": EmbeddingSpec(
        name="gte_multi",
        model_name="Alibaba-NLP/gte-multilingual-base",
        dimension=768,
        max_tokens=8192,
        normalize=True,
        trust_remote_code=True,
    ),

    "qwen3_06b": EmbeddingSpec(
        name="qwen3_06b",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        dimension=1024,
        max_tokens=32768,
        normalize=True,
    ),

    "jina_v4": EmbeddingSpec(
        name="jina_v4",
        model_name="jinaai/jina-embeddings-v4",
        dimension=2048,
        max_tokens=32768,
        normalize=True,
        trust_remote_code=True,
        supports_late_chunking=True,
    ),
}
```

**Agent note:** trước khi chạy benchmark thật, verify lại metadata model theo model card/version được pin trong environment. Không hardcode theo trí nhớ nếu package/model version thay đổi.

---

# 19. Base Embedder

Tạo `src/embeddings/base.py`.

```python
from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedder(ABC):
    @abstractmethod
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def encode_queries(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def token_length(self, text: str) -> int:
        raise NotImplementedError
```

Không được dùng chung query/passage encode một cách mù quáng nếu model yêu cầu instruction/prefix khác nhau.

---

# 20. Hugging Face Embedder

Tạo `src/embeddings/huggingface.py`.

Skeleton:

```python
from sentence_transformers import SentenceTransformer

from src.embeddings.base import BaseEmbedder
from src.embeddings.specs import EmbeddingSpec


class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(self, spec: EmbeddingSpec, batch_size: int = 16, device: str | None = None):
        self.spec = spec
        self.batch_size = batch_size

        self.model = SentenceTransformer(
            spec.model_name,
            trust_remote_code=spec.trust_remote_code,
            device=device,
        )

        self.tokenizer = self.model.tokenizer

    def token_length(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _prepare_passage(self, text: str) -> str:
        if self.spec.passage_instruction:
            return f"{self.spec.passage_instruction}{text}"
        return text

    def _prepare_query(self, text: str) -> str:
        if self.spec.query_instruction:
            return f"{self.spec.query_instruction}{text}"
        return text

    def encode_documents(self, texts: list[str]):
        prepared = [self._prepare_passage(t) for t in texts]
        return self.model.encode(
            prepared,
            normalize_embeddings=self.spec.normalize,
            batch_size=self.batch_size,
            show_progress_bar=True,
        )

    def encode_queries(self, texts: list[str]):
        prepared = [self._prepare_query(t) for t in texts]
        return self.model.encode(
            prepared,
            normalize_embeddings=self.spec.normalize,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
```

---

# 21. Compatibility Guard — bắt buộc

Tạo `src/embeddings/validator.py`.

Mục tiêu: không cho chunk vượt context limit một cách âm thầm.

```python
from dataclasses import dataclass

from src.documents.schema import Chunk
from src.embeddings.base import BaseEmbedder
from src.embeddings.specs import EmbeddingSpec


@dataclass
class ChunkValidationResult:
    chunk_id: str
    token_length: int
    max_tokens: int
    compatible: bool


def validate_chunks(
    chunks: list[Chunk],
    embedder: BaseEmbedder,
    spec: EmbeddingSpec,
) -> list[ChunkValidationResult]:
    results = []

    for chunk in chunks:
        n_tokens = embedder.token_length(chunk.text)
        compatible = n_tokens <= spec.max_tokens

        results.append(
            ChunkValidationResult(
                chunk_id=chunk.chunk_id,
                token_length=n_tokens,
                max_tokens=spec.max_tokens,
                compatible=compatible,
            )
        )

    return results
```

Pipeline mặc định:

```text
Nếu bất kỳ chunk nào incompatible -> experiment FAIL.
```

Không dùng:

```python
truncation=True
```

một cách im lặng.

Nếu sau này cần production fallback, phải là policy explicit:

```yaml
oversize_policy: fail
```

hoặc:

```yaml
oversize_policy: recursive_resplit
```

Nhưng benchmark paper mặc định là `fail` để tránh thay đổi dữ liệu ngoài ý muốn.

---

# 22. Metadata bắt buộc cho mỗi embedding point

Mỗi point Qdrant cần có ít nhất:

```json
{
  "experiment_id": "semantic__bge_m3__512__seed42",
  "document_id": "QD_001",
  "chunk_id": "QD_001__semantic__000003",
  "chunk_method": "semantic",
  "embedding_model": "bge_m3",
  "embedding_dimension": 1024,
  "text": "...",
  "char_length": 3150,
  "word_length": 712,
  "token_length": 1022,
  "embedded_tokens": 1022,
  "was_truncated": false,
  "page_start": 2,
  "page_end": 3,
  "parent_id": null,
  "section_path": ["Chương II", "Điều 5", "Khoản 2"]
}
```

Không được thiếu:

```text
chunk_method
embedding_model
token_length
was_truncated
experiment_id
```

---

# 23. Qdrant collection strategy

Trong giai đoạn research:

```text
1 experiment = 1 collection
```

Naming convention:

```text
rag__{chunker}__{embedding}__{length_or_dynamic}__{run_id}
```

Ví dụ:

```text
rag__fixed__bge_m3__512__r001
rag__recursive__bge_m3__512__r001
rag__semantic__qwen3_06b__dynamic__r001
rag__sahc__jina_v4__dynamic__r001
```

Không trộn các embedding spaces khác nhau trong cùng một vector field khi làm benchmark độc lập.

---

# 24. Qdrant store

Tạo `src/vectorstores/qdrant_store.py`.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantVectorStore:
    def __init__(self, client: QdrantClient):
        self.client = client

    def recreate_collection(self, collection_name: str, vector_size: int):
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def upload(self, collection_name: str, chunks, embeddings, payload_builder):
        points = []

        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=idx,
                    vector=vector.tolist(),
                    payload=payload_builder(chunk),
                )
            )

        self.client.upsert(
            collection_name=collection_name,
            points=points,
        )
```

Production improvement later:

- stable UUID instead of sequential integer
- batching
- retry
- async upload
- payload indexes
- collection aliases

---

# 25. Chunk artifacts phải được lưu riêng

Không được chunk lại mỗi lần đổi embedding model.

Directory convention:

```text
artifacts/chunks/
  fixed__512__64/
  recursive__512__64/
  semantic__p90__max1024/
  sahc__max2048/
```

Mỗi run chunking nên lưu:

```text
chunks.jsonl
stats.json
config.snapshot.yaml
manifest.json
```

Example `chunks.jsonl`:

```json
{"chunk_id":"...","document_id":"...","text":"...","chunk_method":"fixed","metadata":{...}}
```

---

# 26. Chunk statistics

Tạo module tính thống kê ngay sau chunking.

Metrics bắt buộc:

```text
chunk_count
mean_tokens
median_tokens
std_tokens
min_tokens
max_tokens
p50_tokens
p75_tokens
p90_tokens
p95_tokens
p99_tokens
```

Example:

```python
import numpy as np


def compute_chunk_stats(chunks, tokenizer):
    lengths = [
        len(tokenizer.encode(chunk.text, add_special_tokens=False))
        for chunk in chunks
    ]

    if not lengths:
        return {
            "chunk_count": 0,
        }

    arr = np.asarray(lengths, dtype=np.float64)

    return {
        "chunk_count": len(lengths),
        "mean_tokens": float(np.mean(arr)),
        "median_tokens": float(np.median(arr)),
        "std_tokens": float(np.std(arr)),
        "min_tokens": int(np.min(arr)),
        "max_tokens": int(np.max(arr)),
        "p50_tokens": float(np.percentile(arr, 50)),
        "p75_tokens": float(np.percentile(arr, 75)),
        "p90_tokens": float(np.percentile(arr, 90)),
        "p95_tokens": float(np.percentile(arr, 95)),
        "p99_tokens": float(np.percentile(arr, 99)),
    }
```

---

# 27. Experiment configuration

Tạo `configs/experiments.yaml`.

```yaml
seed: 42

chunkers:
  - fixed
  - recursive
  - semantic
  - sahc

embedders:
  - bge_m3
  - gte_multi
  - qwen3_06b
  - jina_v4

fixed_lengths:
  - 256
  - 512
  - 1024
  - 2048

recursive_lengths:
  - 256
  - 512
  - 1024
  - 2048

semantic_profiles:
  - name: semantic_512
    min_tokens: 128
    target_tokens: 512
    max_tokens: 768

  - name: semantic_1024
    min_tokens: 256
    target_tokens: 1024
    max_tokens: 1536

sahc_profiles:
  - name: sahc_512
    max_tokens: 768

  - name: sahc_1024
    max_tokens: 1536

retrieval:
  top_k:
    - 1
    - 3
    - 5
    - 10

qdrant:
  distance: cosine

oversize_policy: fail
```

---

# 28. Experiment ID

Experiment ID phải deterministic hoặc ít nhất reproducible.

Format đề xuất:

```text
{chunker}__{chunk_profile}__{embedding_model}__seed{seed}
```

Ví dụ:

```text
fixed__512__bge_m3__seed42
semantic__semantic_1024__qwen3_06b__seed42
sahc__sahc_1024__jina_v4__seed42
```

---

# 29. Experiment Runner

Tạo `src/pipeline/experiment.py`.

```python
class ExperimentRunner:
    def __init__(
        self,
        chunker_factory,
        embedder_factory,
        vector_store,
        evaluator,
    ):
        self.chunker_factory = chunker_factory
        self.embedder_factory = embedder_factory
        self.vector_store = vector_store
        self.evaluator = evaluator

    def run(self, experiment_config, documents, gold_dataset):
        # 1. resolve experiment id
        # 2. load/create chunk artifacts
        # 3. create embedder
        # 4. validate chunk lengths against embedding context
        # 5. embed passages
        # 6. create Qdrant collection
        # 7. upload vectors + payload
        # 8. encode queries
        # 9. retrieve top-k
        # 10. calculate metrics
        # 11. save metrics + efficiency stats
        # 12. return structured ExperimentResult
        raise NotImplementedError
```

Tuyệt đối không viết một function khổng lồ làm tất cả logic. Tách pipeline stages thành module nhỏ.

---

# 30. Golden evaluation dataset

Tạo `data/gold/queries.jsonl`.

Schema đề xuất:

```json
{
  "query_id": "Q001",
  "question": "Thời hạn thanh toán của Bên B là bao nhiêu?",
  "relevant_document_ids": ["HD_003"],
  "relevant_chunk_ids": [],
  "relevant_section_paths": ["Điều 5/Khoản 2"],
  "reference_text": "Bên B thanh toán trong vòng 30 ngày..."
}
```

Khuyến nghị đánh giá relevance ở nhiều mức:

```text
document-level
section-level
chunk-level
```

Vì dynamic chunking tạo boundary khác nhau, nếu chỉ annotate exact chunk ID theo một splitter sẽ không fair.

Ưu tiên relevance annotation bằng:

```text
document_id + section_path + answer/reference span
```

Sau đó map retrieved chunk vào gold span bằng overlap hoặc structural metadata.

---

# 31. Retrieval metrics

Tạo `src/evaluation/metrics.py`.

Các metric tối thiểu:

```text
HitRate@K
Recall@K
MRR
nDCG@K
MAP
```

## 31.1 HitRate@K

```text
1 nếu top-K chứa ít nhất một relevant item, ngược lại 0.
```

## 31.2 Recall@K

```text
# relevant retrieved in top-K / total relevant
```

## 31.3 MRR

```text
MRR = mean(1 / rank_of_first_relevant)
```

## 31.4 nDCG@K

Dùng graded relevance nếu có section/chunk overlap score.

## 31.5 MAP

Tính Average Precision cho từng query rồi trung bình.

---

# 32. Efficiency metrics

Mỗi experiment phải lưu:

```text
chunk_count
avg_chunk_tokens
median_chunk_tokens
p95_chunk_tokens
max_chunk_tokens

embedding_wall_time_sec
embedding_tokens_per_sec
embedding_chunks_per_sec

qdrant_index_time_sec
qdrant_point_count

query_latency_p50_ms
query_latency_p95_ms
query_latency_p99_ms

index_storage_bytes nếu đo được
```

Quality không phải mục tiêu duy nhất. Có thể tồn tại trường hợp:

```text
Model A:
Recall@10 = 0.960
p95 latency = 220 ms

Model B:
Recall@10 = 0.949
p95 latency = 65 ms
```

Production có thể ưu tiên Model B.

---

# 33. Output results table

Mỗi experiment output một row dạng:

```json
{
  "experiment_id": "sahc__1024__bge_m3__seed42",
  "chunker": "sahc",
  "embedding_model": "bge_m3",
  "chunk_profile": "1024",
  "chunk_count": 12421,
  "mean_tokens": 724.4,
  "p95_tokens": 1822.0,
  "recall_at_5": 0.94,
  "recall_at_10": 0.97,
  "mrr": 0.86,
  "ndcg_at_10": 0.90,
  "map": 0.84,
  "query_p95_ms": 62.3
}
```

Tổng hợp thành CSV/Parquet:

```text
artifacts/reports/experiment_results.csv
artifacts/reports/experiment_results.parquet
```

---

# 34. Fairness requirements cho paper

Nếu chạy:

```text
Fixed avg = 512 tokens
Semantic avg = 900 tokens
SAHC avg = 1800 tokens
```

và SAHC tốt hơn, không thể kết luận chắc chắn improvement đến từ structure-aware boundary.

Reviewer có thể lập luận improvement đến từ nhiều context hơn.

Vì vậy bắt buộc có:

## Experiment A — Controlled token budget

Chunk distributions càng tương đương càng tốt.

Ví dụ:

```text
Fixed target 512
Recursive target 512
Semantic target 512, max 768
SAHC target 512, max 768
```

## Experiment B — Natural chunking

Cho mỗi phương pháp chạy cấu hình tối ưu tự nhiên.

## Experiment C — Length sensitivity

Ví dụ:

```text
256
512
1024
2048
```

Mục tiêu kiểm tra phương pháp có ổn định khi thay token budget hay không.

## Experiment D — Embedding sensitivity

Giữ chunk artifacts cố định, đổi embedding model.

Mục tiêu kiểm tra SAHC có phụ thuộc một model cụ thể hay không.

---

# 35. Late Chunking — Phase 2, không đưa vào baseline đầu tiên

Sau khi tìm được 1–2 tổ hợp tốt nhất, thêm ablation:

```text
SAHC + Embedding Normal
vs
SAHC + Late Chunking
```

hoặc:

```text
Recursive + Embedding Normal
vs
Recursive + Late Chunking
```

Nhờ vậy đóng góp của Late Chunking được tách riêng khỏi boundary strategy.

API đề xuất sau này:

```yaml
embedding:
  model: jina_v4
  mode: late_chunking
```

Không thêm `late_chunking` vào `CHUNKER_REGISTRY`.

---

# 36. Recommended execution phases

## Phase 0 — Environment setup

Agent phải:

1. Tạo project structure.
2. Tạo `pyproject.toml` hoặc `requirements.txt`.
3. Pin dependency versions.
4. Thiết lập logging.
5. Thiết lập YAML config loader.
6. Thiết lập seed.
7. Tạo unit test infrastructure.

Acceptance criteria:

```bash
pytest -q
```

chạy được dù một số tests ban đầu còn placeholder.

---

## Phase 1 — Document schema + loader + normalizer

Agent thực hiện:

1. Common Document schema.
2. Common Chunk schema.
3. Loader cho dữ liệu OCR hiện tại.
4. Normalizer tối thiểu:
   - normalize whitespace
   - preserve paragraph boundaries
   - remove obvious OCR garbage nếu rule an toàn
   - không rewrite nội dung semantic
5. Lưu normalized documents.

Acceptance criteria:

- Có thể load toàn bộ dataset.
- Document IDs ổn định.
- Không mất text.
- Log số document và tổng token.

---

## Phase 2 — Fixed Chunker

Agent thực hiện:

1. FixedTokenChunker.
2. Registry.
3. CLI `run_chunking.py --method fixed`.
4. Save chunk artifacts.
5. Save chunk statistics.
6. Unit tests.

Acceptance criteria:

```bash
python scripts/run_chunking.py --method fixed --chunk-size 512 --overlap 64
```

sinh được:

```text
chunks.jsonl
stats.json
config.snapshot.yaml
```

---

## Phase 3 — Recursive Chunker

Agent thực hiện:

1. RecursiveTokenChunker.
2. Generic separators.
3. Token-based length function.
4. Unit tests.
5. CLI switch bằng config.

Acceptance criteria:

```bash
python scripts/run_chunking.py --method recursive --chunk-size 512 --overlap 64
```

---

## Phase 4 — Semantic Chunker

Agent thực hiện:

1. Atomic unit splitter.
2. Context window builder.
3. Semantic embeddings.
4. Cosine distance.
5. Percentile breakpoint.
6. Min/target/max token guard.
7. Save semantic diagnostic metadata.
8. Unit tests bằng synthetic documents.

Acceptance criteria:

- Topic shift rõ rệt sinh boundary.
- Không chunk nào vượt configured max token sau fallback.
- Kết quả deterministic khi seed/model cố định.

---

## Phase 5 — SAHC Adapter

Agent thực hiện:

1. Tích hợp SAHC hiện tại qua adapter.
2. Không phá code SAHC hiện có.
3. Map output về Common Chunk schema.
4. Preserve:
   - section_path
   - parent_id
   - level
   - page information
5. Oversize fallback.
6. Tests.

Acceptance criteria:

- SAHC trở thành một chunker có thể switch bằng config.
- Pipeline downstream không cần biết chunk được tạo bởi SAHC.

---

## Phase 6 — Embedding Registry

Agent thực hiện:

1. EmbeddingSpec.
2. Model registry.
3. HuggingFaceEmbedder.
4. Query/document encoding abstraction.
5. Model context validator.
6. Tests.

Acceptance criteria:

```bash
python scripts/run_embedding.py \
  --chunks artifacts/chunks/fixed__512__64/chunks.jsonl \
  --model bge_m3
```

- tạo embeddings.
- fail nếu chunk vượt context.
- không silent truncation.

---

## Phase 7 — Qdrant Integration

Agent thực hiện:

1. Qdrant client config.
2. Collection naming.
3. Recreate/create collection option.
4. Batch upsert.
5. Payload metadata.
6. Search wrapper.

Acceptance criteria:

- Có thể index một chunk artifact + embedding model.
- Có thể query thử và trả top-k chunks.
- Payload chứa full experiment metadata.

---

## Phase 8 — Golden dataset + evaluator

Agent thực hiện:

1. Gold schema.
2. Load gold queries.
3. Encode query bằng đúng embedding model.
4. Search cùng collection.
5. Determine relevance.
6. Compute metrics.
7. Save per-query results.
8. Save aggregate results.

Acceptance criteria:

- Có per-query CSV/JSONL.
- Có aggregate metrics.
- Unit tests cho metric formulas.

---

## Phase 9 — Experiment Matrix Runner

Agent thực hiện:

1. Parse `experiments.yaml`.
2. Generate compatible experiment combinations.
3. Skip incompatible chunk_length/model combinations trước khi compute.
4. Reuse chunk artifacts.
5. Reuse embeddings nếu fingerprint giống nhau.
6. Run Qdrant indexing + evaluation.
7. Append result table.

Pseudo-code:

```python
for chunk_profile in chunk_profiles:
    chunks = get_or_create_chunks(chunk_profile)

    for embedding_name in embedding_models:
        spec = embedding_registry[embedding_name]

        if not compatible(chunks, spec):
            mark_incompatible()
            continue

        embeddings = get_or_create_embeddings(chunks, embedding_name)
        collection = get_or_create_qdrant_collection(...)
        metrics = evaluate(...)
        save_result(...)
```

---

## Phase 10 — Reporting

Agent thực hiện:

1. Generate `experiment_results.csv`.
2. Rank experiments theo:
   - Recall@10
   - MRR
   - nDCG@10
3. Tạo Pareto analysis giữa quality và latency.
4. Tạo bảng grouped by embedding model.
5. Tạo bảng grouped by chunker.
6. Tạo bảng grouped by chunk length.

Không kết luận chỉ dựa trên một metric.

---

# 37. CLI design

## Chunking

```bash
python scripts/run_chunking.py \
  --method fixed \
  --chunk-size 512 \
  --overlap 64
```

## Embedding

```bash
python scripts/run_embedding.py \
  --chunk-artifact artifacts/chunks/fixed__512__64/chunks.jsonl \
  --model bge_m3
```

## Indexing

```bash
python scripts/run_indexing.py \
  --experiment fixed__512__bge_m3__seed42
```

## Evaluation

```bash
python scripts/run_evaluation.py \
  --experiment fixed__512__bge_m3__seed42
```

## Full matrix

```bash
python scripts/run_experiments.py \
  --config configs/experiments.yaml
```

---

# 38. Caching strategy

Hash các input để tránh chạy lại.

Chunk artifact fingerprint:

```text
SHA256(
    normalized_dataset_hash
    + chunker_name
    + chunker_config
    + tokenizer_name
)
```

Embedding artifact fingerprint:

```text
SHA256(
    chunk_artifact_hash
    + embedding_model_name
    + model_revision
    + embedding_config
)
```

Nếu fingerprint tồn tại thì reuse.

---

# 39. Reproducibility requirements

Mỗi run phải lưu:

```text
experiment_id
UTC/local timestamp
seed
git commit
python version
dependency versions
CUDA version
GPU name
chunk config
embedding model name
embedding model revision
Qdrant version
metric config
```

Tạo file:

```text
artifacts/reports/{experiment_id}/manifest.json
```

---

# 40. Logging requirements

Mỗi stage log:

```text
number of documents
number of chunks
token distribution
model name
model max tokens
embedding dimension
batch size
embedding time
Qdrant collection
query count
retrieval latency
metrics
```

Không log full document content mặc định nếu dữ liệu nhạy cảm.

---

# 41. Error handling

Các lỗi sau phải fail rõ ràng:

```text
Unknown chunker
Unknown embedding model
Chunk > embedding max tokens
Embedding dimension mismatch with Qdrant
Missing gold dataset
Empty document set
Duplicate chunk ID
NaN/Inf embedding
Qdrant upload failure
```

Không được swallow exception rồi tiếp tục benchmark như không có lỗi.

---

# 42. Tests tối thiểu

## Chunker tests

```text
fixed
recursive
semantic
sahc adapter
```

## Embedding tests

```text
model registry
query/document prefix
max token validator
normalize flag
embedding shape
```

## Qdrant tests

Có thể dùng local/in-memory test setup nếu phù hợp.

Test:

```text
create collection
insert point
search point
payload round-trip
```

## Metric tests

Dùng hand-calculated examples để verify:

```text
Recall@K
HitRate@K
MRR
nDCG@K
MAP
```

---

# 43. Recommended first experiment matrix

Không chạy toàn bộ 64 experiments ngay lập tức.

## Stage 1 — Smoke test

```text
Chunkers:
- fixed
- recursive

Embedding:
- bge_m3

Lengths:
- 512
```

Tổng 2 experiments.

## Stage 2 — Chunker validation

```text
Chunkers:
- fixed
- recursive
- semantic
- sahc

Embedding:
- bge_m3

Length profile:
- ~512
```

Tổng 4 experiments.

Mục tiêu: verify toàn pipeline.

## Stage 3 — Embedding sensitivity

```text
Chunkers:
- fixed
- recursive
- semantic
- sahc

Embedding:
- bge_m3
- gte_multi
- qwen3_06b
- jina_v4

Length profile:
- ~512
```

Tổng 16 experiments.

## Stage 4 — Length sensitivity

Chỉ lấy 2 chunker tốt nhất và 2 embedding tốt nhất.

```text
Lengths:
256
512
1024
2048
```

Tổng:

```text
2 × 2 × 4 = 16 experiments
```

Cách này tiết kiệm compute nhưng vẫn cung cấp evidence tốt.

---

# 44. Recommended model-selection interpretation

Không chọn model chỉ dựa vào maximum context.

Đánh giá:

```text
Retrieval quality
Latency
GPU memory
Embedding throughput
Storage cost
Operational complexity
```

Output cuối có thể là:

```text
Best quality:
SAHC + Jina

Best quality/cost:
SAHC + BGE-M3

Best low-latency:
Recursive + GTE
```

---

# 45. Tìm chunk length tối ưu

Không chọn chunk dài nhất một cách mặc định.

Phương pháp:

1. Với mỗi chunker, tạo profile length.
2. Với mỗi embedding model, loại các profile incompatible.
3. Chạy retrieval evaluation.
4. Vẽ hoặc tổng hợp:

```text
chunk length -> Recall@K
chunk length -> MRR
chunk length -> nDCG
chunk length -> latency
```

5. Tìm điểm tốt nhất hoặc Pareto-optimal.

Khái niệm cuối cùng:

```text
BestConfig = argmax Quality(chunker, embedding, length)
```

subject to:

```text
length <= embedding_context_limit
latency <= production_SLA
memory <= infrastructure_limit
```

---

# 46. Late Chunking extension

Chỉ thực hiện sau baseline.

Interface đề xuất:

```python
class BaseEmbeddingStrategy:
    def encode_chunks(self, document, chunks):
        ...
```

Implement:

```text
NormalEmbeddingStrategy
LateChunkingEmbeddingStrategy
```

Late Chunking pipeline:

```text
Full document
    |
    v
Transformer token embeddings
    |
    v
Chunk spans
    |
    v
Pooling per span
    |
    v
Chunk vectors
```

Không thay đổi chunk boundary.

Ablation:

```text
SAHC + BGE normal
vs
SAHC + BGE late  # nếu model/API hỗ trợ đúng

SAHC + Jina normal
vs
SAHC + Jina late
```

---

# 47. Parent-child / Hierarchical Retrieval extension

Sau flat retrieval benchmark, thêm:

```text
retrieve child
    |
    v
resolve parent_id
    |
    v
expand parent context
    |
    v
LLM
```

Ablation:

```text
SAHC flat retrieval
vs
SAHC parent-child retrieval
```

Không trộn contribution này vào chunking baseline ban đầu.

---

# 48. Hybrid retrieval extension

BGE-M3 có thể phù hợp cho nghiên cứu dense/sparse/hybrid.

Phase sau:

```text
Dense retrieval
Sparse retrieval
Hybrid retrieval
Hybrid + reranker
```

Nhưng phải giữ chunk artifacts giống nhau khi so sánh retrieval strategy.

---

# 49. Agent execution contract

Agent/Codex phải thực hiện từng phase theo thứ tự.

Không được:

1. Viết toàn bộ project trong một file.
2. Hardcode chunker trong main pipeline.
3. Hardcode model trong embedding pipeline.
4. Silent truncate.
5. Re-chunk mỗi khi đổi embedding model.
6. Trộn vector của các model khác nhau mà không có named-vector design rõ ràng.
7. Thay đổi gold dataset giữa experiments.
8. Dùng semantic model khác nhau giữa runs mà không log.
9. Dùng SAHC-specific metadata làm lợi thế retrieval trong fair baseline nếu baseline khác không có metadata tương đương, trừ khi đây là experiment structure-aware được khai báo rõ.

Agent phải:

1. Implement module nhỏ.
2. Viết unit test ngay sau mỗi module.
3. Chạy test trước khi chuyển phase.
4. Log output.
5. Snapshot config.
6. Không phá API các phase đã hoàn thành.

---

# 50. Agent task sequence

## TASK-001 — Scaffold project

**Goal:** tạo toàn bộ directory structure và dependency setup.

**Deliverables:**

```text
project tree
pyproject.toml
requirements.txt nếu cần
README.md
logging config
config loader
```

**Acceptance:**

```bash
pytest -q
```

khởi chạy thành công.

---

## TASK-002 — Common schemas

**Goal:** implement Document, Chunk, serialization JSONL.

**Deliverables:**

```text
src/documents/schema.py
src/documents/loader.py
```

**Acceptance:** serialize -> deserialize không mất metadata.

---

## TASK-003 — OCR normalizer

**Goal:** tạo deterministic normalization.

**Rules:**

- normalize Unicode nếu cần
- normalize duplicate spaces
- preserve meaningful newlines
- không paraphrase nội dung
- không dùng LLM normalization trong baseline

**Acceptance:** output deterministic.

---

## TASK-004 — Fixed chunker

Implement + tests + CLI.

---

## TASK-005 — Recursive chunker

Implement + tests + CLI.

---

## TASK-006 — Semantic chunker

Implement atomic units, embeddings, breakpoint, min/max enforcement, diagnostics, tests.

---

## TASK-007 — SAHC adapter

Wrap SAHC hiện tại vào common interface.

---

## TASK-008 — Chunk artifact manager

Implement caching, manifest, stats.

---

## TASK-009 — Embedding specs + registry

Implement model metadata and factory.

---

## TASK-010 — Hugging Face embedder

Implement query/document encoding.

---

## TASK-011 — Compatibility guard

Fail oversize chunks.

---

## TASK-012 — Embedding artifact manager

Cache embeddings by fingerprint.

---

## TASK-013 — Qdrant store

Create collection, upsert, query, payload.

---

## TASK-014 — Gold dataset schema

Implement query relevance representation.

---

## TASK-015 — Metrics

Implement HitRate@K, Recall@K, MRR, nDCG@K, MAP with unit tests.

---

## TASK-016 — Retrieval evaluator

Run queries and save per-query results.

---

## TASK-017 — Experiment runner

Generate experiment matrix and execute compatible combinations.

---

## TASK-018 — Reporting

CSV/Parquet aggregation + ranking + Pareto summary.

---

## TASK-019 — Late Chunking ablation

Chỉ bắt đầu khi TASK-001 đến TASK-018 ổn định.

---

## TASK-020 — Hierarchical retrieval ablation

Child retrieval + parent expansion.

---

# 51. Definition of Done

Framework được coi là hoàn thành phase baseline khi:

1. Có 4 chunker switch bằng YAML.
2. Có ít nhất 4 embedding model switch bằng YAML.
3. Có chunk artifact caching.
4. Có embedding caching.
5. Không silent truncation.
6. Có Qdrant collection per experiment.
7. Có gold query dataset loader.
8. Có Recall@K, HitRate@K, MRR, nDCG@K, MAP.
9. Có efficiency metrics.
10. Có experiment runner.
11. Có result table tổng hợp.
12. Có tests cho core modules.
13. Có reproducibility manifest.
14. Có controlled-token-budget benchmark.
15. Có natural-length benchmark.

---

# 52. Recommended initial configuration

```yaml
chunkers:
  - fixed
  - recursive
  - semantic
  - sahc

embedders:
  - bge_m3
  - gte_multi
  - qwen3_06b
  - jina_v4

controlled_chunk_lengths:
  - 512
  - 1024
  - 2048

retrieval:
  top_k:
    - 1
    - 3
    - 5
    - 10

vector_db:
  type: qdrant
  distance: cosine

oversize_policy: fail

seed: 42
```

---

# 53. Recommended baseline interpretation

Vòng đầu tiên:

```text
Fixed / Recursive / Semantic / SAHC
                    ×
                  BGE-M3
                    ×
               ~512 tokens
```

Nếu pipeline ổn định, mở rộng:

```text
4 Chunkers
    ×
4 Embedding Models
    ×
1 controlled length profile
```

Sau đó mới làm length sensitivity với top-2 chunker và top-2 model.

Không nên chạy full combinatorial search ngay từ đầu nếu chưa verify evaluator và gold labels.

---

# 54. Kết quả nghiên cứu mong muốn

Sau toàn bộ benchmark, framework phải trả lời được các câu hỏi:

1. Chunking method nào tốt nhất khi dùng cùng embedding model?
2. Embedding model nào ổn định nhất trên nhiều chunking methods?
3. SAHC có cải thiện retrieval so với Fixed/Recursive/Semantic không?
4. Improvement của SAHC còn giữ khi kiểm soát token budget không?
5. Chunk length tối ưu cho mỗi embedding model là bao nhiêu?
6. Long-context model có thực sự hưởng lợi từ long chunk không?
7. Model tốt nhất về quality có còn tốt khi xét latency/storage không?
8. Late Chunking có tạo thêm cải thiện sau khi boundary đã tối ưu không?
9. Parent-child retrieval có cải thiện answer context mà không làm giảm precision không?

---

# 55. Final target architecture

```text
                    EXPERIMENT ENGINE
                           |
       +-------------------+-------------------+
       |                                       |
       v                                       v
 CHUNKING ENGINE                        EMBEDDING ENGINE
       |                                       |
       +-- Fixed                               +-- BGE-M3
       +-- Recursive                           +-- GTE-multilingual
       +-- Semantic                            +-- Qwen3-Embedding
       +-- SAHC                                +-- Jina-v4
       |                                       |
       +-------------------+-------------------+
                           |
                           v
                  Compatibility Guard
                           |
             chunk_tokens <= model_limit
                           |
                           v
                     Vectorization
                           |
                           v
                        Qdrant
                           |
                           v
                  Retrieval Evaluator
                           |
        +------------------+-------------------+
        |                  |                   |
        v                  v                   v
    Recall@K              MRR                nDCG
        |                  |                   |
        +------------------+-------------------+
                           |
                           v
                  Experiment Results
                           |
                           v
          Best Chunker × Model × Length
```

Mục tiêu tối ưu:

```text
Best(chunker, embedding, length)
    = argmax RetrievalQuality
```

subject to:

```text
chunk_length <= embedding_context_limit
latency <= production_SLA
memory <= infrastructure_limit
```

---

# 56. Chỉ dẫn ngắn cho Agent/Codex trước khi bắt đầu

Agent phải đọc toàn bộ tài liệu này trước khi code.

Thứ tự bắt buộc:

```text
TASK-001
  -> TASK-002
  -> TASK-003
  -> TASK-004
  -> TASK-005
  -> TASK-006
  -> TASK-007
  -> TASK-008
  -> TASK-009
  -> TASK-010
  -> TASK-011
  -> TASK-012
  -> TASK-013
  -> TASK-014
  -> TASK-015
  -> TASK-016
  -> TASK-017
  -> TASK-018
```

Sau mỗi task:

1. Chạy unit tests liên quan.
2. Không chuyển task nếu test fail.
3. Ghi lại thay đổi vào changelog hoặc task log.
4. Không sửa API đã public nếu không thực sự cần thiết.
5. Nếu phải thay đổi spec, ghi rõ lý do và impact.

Late Chunking và hierarchical retrieval chỉ được thực hiện sau khi baseline framework ổn định.

---

# 57. Prompt gợi ý để giao cho coding agent

```text
You are a Senior Python Engineer and RAG Evaluation Engineer.

Read RAG_CHUNKING_EMBEDDING_EXPERIMENT_SPEC.md completely before changing code.

Implement the project strictly phase-by-phase according to TASK-001 through TASK-018.

Rules:
1. Do not implement multiple phases in a single monolithic module.
2. Use clean interfaces and registries for chunkers and embedding models.
3. Preserve deterministic behavior and experiment reproducibility.
4. Never silently truncate chunks before embedding.
5. Reuse chunk artifacts across embedding experiments.
6. Store each experiment in an independent Qdrant collection during research.
7. Write unit tests for every core algorithmic component.
8. After each task, run the related tests and report changed files, test results, and any deviations from the specification.
9. Do not start Late Chunking or hierarchical retrieval until the baseline experiment framework is complete and passing tests.
10. Prefer small, reviewable commits/changes.

Start with TASK-001 only. Complete it, run tests, summarize the result, then proceed to TASK-002.
```

---

# 58. Kết luận

Thiết kế này tách rõ bốn biến quan trọng của hệ thống RAG:

```text
Chunk Boundary
Embedding Model
Chunk Length
Retrieval Strategy
```

Nhờ vậy có thể đánh giá khoa học hơn và tránh kết luận sai do thay đổi nhiều thành phần cùng lúc.

Đối với nghiên cứu SAHC, baseline quan trọng nhất là:

```text
Fixed
Recursive
Semantic
SAHC
```

sau đó kiểm tra sensitivity trên nhiều embedding model.

Late Chunking được xem là contextual embedding strategy và được đánh giá sau bằng ablation riêng.

Hierarchical/parent-child retrieval cũng được đánh giá riêng sau flat retrieval baseline.

Framework cuối cùng phải cho phép trả lời không chỉ:

> Chunking nào tốt nhất?

mà chính xác hơn:

> Với corpus OCR tiếng Việt này, tổ hợp Chunking Method × Embedding Model × Chunk Length nào đạt retrieval quality tốt nhất dưới ràng buộc latency, storage và model context?

