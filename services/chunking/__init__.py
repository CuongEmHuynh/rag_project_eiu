"""Public API cho Structure-Aware Hierarchical Chunking v2 (SAHC-v2)."""

from .chunk_builder import (
    build_chunks_from_ocr_document,
    build_document_chunks,
    build_document_chunks_v2,
    create_atomic_units,
)
from .legacy import chunk_legal_document_v1
from .models import (
    AtomicUnit,
    Chunk,
    ChunkingConfig,
    DocumentNode,
    DocumentTree,
    OCRBlock,
    OCRDocument,
    PackedUnit,
    ParsedTable,
    RetrievalResult,
)
from .retrieval_text import build_retrieval_text, chunk_to_payload
from .validators import ChunkValidationError, validate_chunks

__all__ = [
    "AtomicUnit",
    "Chunk",
    "ChunkValidationError",
    "ChunkingConfig",
    "DocumentNode",
    "DocumentTree",
    "OCRBlock",
    "OCRDocument",
    "PackedUnit",
    "ParsedTable",
    "RetrievalResult",
    "build_chunks_from_ocr_document",
    "build_document_chunks",
    "build_document_chunks_v2",
    "build_retrieval_text",
    "chunk_legal_document_v1",
    "chunk_to_payload",
    "create_atomic_units",
    "validate_chunks",
]

