import csv
from qdrant_client import QdrantClient,models
import re
from sentence_transformers import SentenceTransformer
import re
import uuid
from pathlib import Path
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer


# Qdrant Config
SERVERQDRANT="http://222.255.214.30:6333"
COLLECTION_NAME="rag_document"
MODEL_EMBEDDING="bkai-foundation-models/vietnamese-bi-encoder"


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
    

def chunk_legal_document(text: str):
    chunks = []

    # Điều x:
    parts = re.split(r"(?=Điều\s+\d+\s*:)", text)

    for p in parts:
        p = p.strip()
        if not p:
            continue

        chunk_type = "dieu" if p.startswith("Điều") else "header"

        # bảng điểm rất dài → cắt mềm
        if len(p) > 3000:
            sub_parts = re.split(r"\n{2,}", p)
            for sp in sub_parts:
                if len(sp.strip()) > 200:
                    chunks.append((chunk_type, sp.strip()))
        else:
            chunks.append((chunk_type, p))

    return chunks



def clean_data(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[?]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def build_embedding_text(meta, chunk_text):
    return f"""
    Văn bản: {meta['Summary']}
    Số: {meta['No']}
    Cơ quan ban hành: {meta['Author']}
    Ngày: {meta['DateDocument']}

    Nội dung:
    {chunk_text}
    """.strip()

def load_data(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        total=0
        for idx, row in enumerate(reader, start=1):
            total+=1
            points = []
            print(f"Processing document {idx} - Total processed: {total}")
            META = {
                "Id": row.get("Id"),
                "KeyFileId": row.get("KeyFileId"),
                "Page": row.get("Page"),
                "No": row.get("No"),
                "Author": row.get("Author"),
                "Summary": row.get("Summary"),
                "DateDocument": row.get("DateDocument"),
                "RecordId": row.get("RecordId"),
                "FileNameMinio": row.get("FileNameMinio"),
                "FilePathMinio": row.get("FilePathMinio")
            }
            ocr_path= f"./data/file_contents/{META['Id']}.txt"
            ocr_text_cleaned = clean_data(open(ocr_path, "r", encoding="utf-8").read())
            chunks = chunk_legal_document(ocr_text_cleaned)
            for idx, (chunk_type, chunk_text) in enumerate(chunks):
                embedding_text = build_embedding_text(META, chunk_text)
                vector = model.encode(embedding_text, normalize_embeddings=True)
                point_id = str(
                    uuid.uuid5(uuid.UUID(META["Id"]), f"chunk-{idx}")
                )

                payload = {
                    **META,
                    "chunk_index": idx,
                    "chunk_type": chunk_type,
                    "chunk_text": chunk_text,
                    "source": "OCR"
                }

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload=payload
                    )
                )
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            del points
            print(f"Inserted document {idx} with {len(chunks)} chunks.")
                
            

def embedding_search(query: str, top_k: int=5):
    query_vector = model.encode(query, normalize_embeddings=True)
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector.tolist(),
        limit=top_k
    )
    return search_result
               


if __name__=="__main__":
    # Create Collection 
    # creat_collection(client)
    # path_file_data= "./data/documents.csv"
    # load_data(path_file_data)
    
    ## serch test 
    
    
    print('Embedding service')