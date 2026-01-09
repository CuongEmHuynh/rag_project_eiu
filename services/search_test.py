import csv
from qdrant_client import QdrantClient
import re
from sentence_transformers import SentenceTransformer
from openai import OpenAI


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

SERVERQDRANT="http://222.255.214.30:6333"
COLLECTION_NAME="rag_document"
MODEL_EMBEDDING="bkai-foundation-models/vietnamese-bi-encoder"

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"   # fake key, LM Studio không check
)


qdrant= QdrantClient(url=SERVERQDRANT)

embedding_model = SentenceTransformer(
    MODEL_EMBEDDING
)


def embed_query(question: str):
    return embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()
    
def retrieve(question, top_k=5, qdrant_filter=None):
    vector = embed_query(question)
    

    return qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=5,
        with_payload=True
    ).points
    
def build_context(results):
    context_blocks = []
    sources = []

    for i, r in enumerate(results, start=1):
        p = r.payload

        context_blocks.append(
            f"[CHUNK {i}]\n{p['chunk_text']}"
        )

        sources.append({
            "chunk": i,
            "file": p["FileNameMinio"],
            "page": p["Page"],
            "no": p["No"]
        })

    return "\n\n".join(context_blocks), sources

def build_prompt(question, context):
    return f"""
    Bạn là trợ lý AI chuyên tra cứu văn bản hành chính – giáo dục.

        QUY TẮC BẮT BUỘC:
        - Chỉ sử dụng thông tin trong CONTEXT
        - Tuyệt đối không suy đoán
        - Nếu không có thông tin → nói rõ
        - Mỗi ý trả lời phải trích dẫn [CHUNK x]

        CONTEXT:
        {context}

        CÂU HỎI:
        {question}

        YÊU CẦU:
        - Trả lời bằng tiếng Việt
        - Ngắn gọn, đúng trọng tâm
        - Có trích dẫn rõ ràng

    """

def call_llm(prompt: str, max_tokens: int = 512):
    
    resp = client.chat.completions.create(
        model="qwen2.5-7b-instruct-1m",
        messages=[
            {"role": "user", 
             "content": prompt}
        ],
        temperature=0.0,
        max_tokens=max_tokens
    )

    print(resp.choices[0].message.content)

def rag(question, top_k=5, qdrant_filter=None):
    results = retrieve(question, top_k, qdrant_filter)

    if not results:
        return "Không tìm thấy thông tin trong tài liệu.", []

    return results
    # context, sources = build_context(results)
    # prompt = build_prompt(question, context)

    # answer = call_llm(prompt)

    # return answer, sources


question = "Quyết định rút môn học của sinh viên Phạm Thị Thanh Hòa"

# answer, sources = rag(question)

results = rag(question)
for r in results:
    print("-" * 80)
    print("Score:", round(r.score, 4))
    print("Điều:", r.payload["chunk_text"])
    

# print("ANSWER:\n", answer)
# print("\nSOURCES:")
# for s in sources:
#     print(s)



