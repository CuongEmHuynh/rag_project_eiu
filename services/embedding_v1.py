import csv
from qdrant_client import QdrantClient,models
import re
from sentence_transformers import SentenceTransformer
import re
import uuid
from pathlib import Path
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Filter, FieldCondition, MatchAny


# Qdrant Config
SERVERQDRANT="http://222.255.214.30:6333"
COLLECTION_NAME="rag_document_v2"
MODEL_EMBEDDING="bkai-foundation-models/vietnamese-bi-encoder"
OCR_DIR = Path("./data/file_contents")
CSV_PATH = "./data/query_data2.csv"   
    
SECTION_RE = re.compile(
    r"(?m)^(?:Chương\s+[IVXLC]+|Mục\s+\d+|Điều\s+\d+)\s*[:\.]?",
    re.IGNORECASE
)
KHOAN_SPLIT_RE = re.compile(r"(?m)(?=^\s*\d+\.\s)")
MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 1200
OVERLAP_WORDS = 40


client= QdrantClient(url=SERVERQDRANT)

model = SentenceTransformer(
    MODEL_EMBEDDING
)
# Create collection in Qdrant
def creat_collection(client):

    vectors_config = models.VectorParams(size=768, distance=models.Distance.COSINE)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=vectors_config
    )


def _split_sections(text: str) -> list[str]:
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [text.strip()]

    sections = []
    preamble = text[:matches[0].start()].strip()
    if preamble:
        sections.append(preamble)

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(text[start:end].strip())

    return sections


def _chunk_by_size(text: str, max_chars: int, overlap_words: int) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    chunks = []
    buf = []
    length = 0

    for line in lines:
        if buf and length + len(line) + 1 > max_chars:
            chunk = " ".join(buf).strip()
            chunks.append(chunk)

            if overlap_words > 0:
                tail_words = chunk.split()[-overlap_words:]
                buf = [" ".join(tail_words)]
                length = len(buf[0])
            else:
                buf = []
                length = 0

        buf.append(line)
        length += len(line) + 1

    if buf:
        chunks.append(" ".join(buf).strip())

    return chunks


def _merge_small_chunks(chunks: list[str], min_chars: int, max_chars: int) -> list[str]:
    merged = []
    for chunk in chunks:
        if not merged:
            merged.append(chunk)
            continue

        if len(chunk) < min_chars and len(merged[-1]) + 1 + len(chunk) <= max_chars:
            merged[-1] = f"{merged[-1].rstrip()} {chunk.lstrip()}"
        else:
            merged.append(chunk)

    return merged


def chunk_legal_document_v2(text: str) -> list[str]:
    text = re.sub(r"\n{2,}", "\n", text).strip()
    sections = _split_sections(text)
    chunks = []

    for section in sections:
        if len(section) <= MAX_CHUNK_CHARS:
            chunks.append(section)
            continue

        # ưu tiên tách theo Khoản nếu có, rồi mới tách theo kích thước
        khoans = re.split(KHOAN_SPLIT_RE, section)
        if len(khoans) > 1:
            for k in khoans:
                k = k.strip()
                if not k:
                    continue
                if len(k) <= MAX_CHUNK_CHARS:
                    chunks.append(k)
                else:
                    chunks.extend(_chunk_by_size(k, MAX_CHUNK_CHARS, OVERLAP_WORDS))
        else:
            chunks.extend(_chunk_by_size(section, MAX_CHUNK_CHARS, OVERLAP_WORDS))

    return _merge_small_chunks(chunks, MIN_CHUNK_CHARS, MAX_CHUNK_CHARS)

def clean_ocr_semantic(text: str) -> str:
    """
    Clean dữ liệu đã OCR, loại bỏ các dòng rác, header/footer lặp
    
    :param text: Description
    :type text: str
    :return: Description
    :rtype: str
    """
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()

        # bỏ dòng rỗng / quá ngắn
        if len(line) < 5:
            continue

        # bỏ header/footer lặp
        if re.search(
            r"CỘNG HÒA XÃ HỘI| TRƯỜNG ĐẠI HỌC| QUỐC TẾ MIỀN ĐÔNG",
            line,
            re.IGNORECASE
        ):
            continue

        # bỏ dòng toàn ký tự rác
        if re.fullmatch(r"[A-Z0-9\s\-]{1,12}", line):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def build_embedding_text(chunk_text: str) -> str:
    """
    Chỉ embed semantic content
    """
    return chunk_text.strip()


def _normalize_object_name(file_path: str, file_name: str) -> str:
    """Build MinIO object path from CSV columns."""
    file_path = (file_path or "").strip()
    file_name = (file_name or "").strip()

    if file_path:
        normalized = file_path.rstrip("/")
        if file_name and not normalized.endswith(file_name):
            normalized = f"{normalized}/{file_name}"
        return normalized.lstrip("/")

    return file_name.lstrip("/").lower()

def load_and_index(csv_path: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for doc_idx, row in enumerate(reader, start=1):
            doc_id = row["Id"]
            ocr_file = OCR_DIR / f"{doc_id.upper()}.txt"

            if not ocr_file.exists():
                print(f"[SKIP] OCR not found for {doc_id}")
                continue

            print(f"Processing document {doc_idx} – {doc_id}")

            raw_text = ocr_file.read_text(encoding="utf-8")
            clean_text = clean_ocr_semantic(raw_text)
            chunks = chunk_legal_document_v2(clean_text)

            points = []
            
            object_column = "FilePathMinio"
            file_name_column = "FileNameMinio"
            file_path = row.get(object_column) or row.get(object_column.lower())
            file_name = row.get(file_name_column) or row.get(file_name_column.lower())
            object_name = _normalize_object_name(file_path, file_name.lower())

            for i, chunk in enumerate(chunks):
                vector = model.encode(
                    build_embedding_text(chunk),
                    normalize_embeddings=True
                )

                point_id = str(
                    uuid.uuid5(uuid.UUID(doc_id), f"chunk-{i}")
                )

                payload = {
                    "doc_id": doc_id,
                    "doc_no": row["No"],
                    "author": row["Author"],
                    "date": row["DateDocument"],
                    "summary": row["Summary"],
                    "file_url":object_name,
                    "chunk_index": i,
                    "chunk_text": chunk,
                    "source": "OCR"
                }

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload=payload
                    )
                )

            if points:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )

            print(f"Inserted {len(points)} chunks.")

def creat_collection(client):

    vectors_config = models.VectorParams(size=768, distance=models.Distance.COSINE)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=vectors_config
    )

def embedding_search(query: str, top_k: int = 5):
    flt = Filter(
        must=[FieldCondition(key="doc_id", match=MatchAny(any=['ef0359f7-e2a8-b9fc-4844-3a18b9689c78']))]
    )
    query_vector = model.encode(query, normalize_embeddings=True)
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=flt,
        limit=top_k
    )
    return search_result.points


if __name__ == "__main__":
    creat_collection(client)
    # # Index dữ liệu
    load_and_index(CSV_PATH)
    print("Indexing completed.")

    # # Test search
   
    # results = embedding_search(
    #     "Các quyết định của sinh viên Phạm Văn Giang",
    #     top_k=5
        
    # )

    # for r in results:
    #     print("-" * 80)
    #     print("Score:", round(r.score, 4))
    #     print("Điều:", r.payload["chunk_text"])
