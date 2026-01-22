import os
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from sentence_transformers import SentenceTransformer

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import lru_cache

# Neo4j connection setup
NEO4J_URI = "bolt://222.255.214.30:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "StrongPassword123!"
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)


# Qdrant config 
SERVERQDRANT="http://222.255.214.30:6333"
COLLECTION_NAME="rag_document_v2"
MODEL_EMBEDDING="bkai-foundation-models/vietnamese-bi-encoder"
qdrant= QdrantClient(url=SERVERQDRANT)
model = SentenceTransformer(
    MODEL_EMBEDDING
)


# Event keyword dictionary => Về mặt lâu dài có thể lưu vào CSDL để dễ bảo trì hơn
EVENT_KEYWORDS = {
    "DISCIPLINARY_ACTION": [
        "kỷ luật", "khiển trách", "cảnh cáo", "đình chỉ"
    ],
    "REWARD": [
        "khen thưởng", "tuyên dương", "khen"
    ],
    "DECISION": [
        "quyết định", "áp dụng", "ban hành"
    ]
}

# Kiểm tra loại sự kiện dựa trên từ khóa trong câu hỏi
def detect_event_type(question: str):
    q = question.lower()
    for event_type, keywords in EVENT_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                return event_type
    return None

# Kiểm tra tên người dựa trên dữ liệu trong Neo4j
def detect_person_from_graph(driver, question: str):
    """
    Detect person name by matching against Person nodes in Neo4j.
    Đây là bước quyết định độ CHÍNH XÁC.
    """
    query = """
    MATCH (p:Person)
    WHERE $q CONTAINS p.name
    RETURN p.name AS name
    ORDER BY size(p.name) DESC
    LIMIT 1
    """
    with driver.session() as session:
        rec = session.run(query, q=question).single()
        return rec["name"] if rec else None


# Định tuyến câu hỏi dựa trên việc phát hiện Person và Event
def route_questions(driver, question: str):
    event_type = detect_event_type(question)
    person = detect_person_from_graph(driver, question)
    if person and event_type:
        return "PERSON_EVENT", person, event_type
    elif person:
        return "PERSON_ONLY", person, None
    elif event_type:
        return "EVENT_ONLY", None, event_type
    else:
        return "VECTOR_ONLY", None, None


# ====== Các hàm hỗ trợ query theo từng loại câu hỏi ======
# == Person & Event ==
def graph_query_person_event(driver, person_name: str, event_type: str):
    print("[Graph Query] Person + Event")
    query = """
    MATCH (p:Person {name:$name})
    <-[:APPLIES_TO]-(e:Event {type:$type})
    <-[:DESCRIBES]-(d:Document)
    RETURN DISTINCT d.id AS doc_id
    """
    
    # query = """
    #     MATCH (d:Document)-[:DESCRIBES]->(e:Event {type:$type})
    #     MATCH (e)-[:APPLIES_TO]->(p:Person {name:$name})
    #     WITH d, e, collect(p) AS persons
    #     WHERE size(persons) = 1
    #     RETURN DISTINCT d.id AS doc_id
    # """
    with driver.session() as session:
        return [r["doc_id"] for r in session.run(
            query, name=person_name, type=event_type
        )]

# == Person ==
def graph_query_person(driver, person_name: str):
    query = """
    MATCH (p:Person {name:$name})
    <-[:APPLIES_TO]-(e:Event)
    <-[:DESCRIBES]-(d:Document)
    RETURN DISTINCT d.id AS doc_id
    """
    with driver.session() as session:
        return [r["doc_id"] for r in session.run(query, name=person_name)]

# == Event ==
def graph_query_event(driver, event_type: str):
    query = """
    MATCH (e:Event {type:$type})
    <-[:DESCRIBES]-(d:Document)
    RETURN DISTINCT d.id AS doc_id
    """
    with driver.session() as session:
        return [r["doc_id"] for r in session.run(query, type=event_type)]
# ================================

# Định tuyến và truy vấn đồ thị tri thức
def graph_retrieve_documents(driver, question: str):
    route, person, event = route_questions(driver, question)

    print(f"[ROUTE] {route} | Person={person} | Event={event}")

    if route == "PERSON_EVENT":
        return graph_query_person_event(driver, person, event)

    elif route == "PERSON_ONLY":
        return graph_query_person(driver, person)

    elif route == "EVENT_ONLY":
        return graph_query_event(driver, event)

    else:
        return [], "VECTOR_ONLY"   # fallback → Vector-first


# ===== Qdrant Vector Search =====
def vector_search_filtered(qdrant, collection, query_vector, allowed_doc_ids, limit=5):
    print(allowed_doc_ids )
    flt = Filter(
        must=[FieldCondition(key="doc_id", match=MatchAny(any=allowed_doc_ids))]
    )
    return qdrant.query_points(
        collection_name=collection,
        query=query_vector,
        query_filter=flt,
        limit=limit
    ).points
    
    
def hybrid_retrieve(
    question: str,
    driver,
    qdrant: QdrantClient,
    embedder,
    collection: str,
    top_k: int = 5
):
    # (1) Graph filter
    allowed_doc_ids = graph_retrieve_documents(driver, question)

    # (2) Embed query
    qvec = embedder.encode(question).tolist()

    # (3) Vector search
    if allowed_doc_ids:
        allowed_doc_ids= [doc_id.lower() for doc_id in allowed_doc_ids]
        hits = vector_search_filtered(
            qdrant, collection, qvec, allowed_doc_ids, limit=top_k
        )
    else:
        hits = qdrant.search(
            collection_name=collection,
            query_vector=qvec,
            limit=top_k
        )
    return hits


def sort_hits_in_order(hits):
    return sorted(
        hits,
        key=lambda h: (
            h.payload.get("doc_id"),
            int(h.payload.get("chunk_index", 0))
        )
    )

def build_context1(hits):
    blocks = []
    hits = sort_hits_in_order(hits)
    for h in hits:
        blocks.append(
            f"{h.payload['chunk_text']}"
        )
        
    blocks.append(
            f"File_URL: {h.payload.get('file_url','')}"
        )
    return "\n".join(blocks)




# ===== Test run with context LLM  model Qwen/Qwen2.5-1.5B-Instruct=====
ANSWER_PROMPT = """
Bạn là trợ lý trả lời dựa trên bằng chứng.

QUY TẮC BẮT BUỘC:
- CHỈ sử dụng thông tin trong CONTEXT.
- KHÔNG suy đoán.
- KHÔNG bổ sung kiến thức bên ngoài.
- Mỗi ý chính PHẢI kèm trích dẫn [DOC_ID].
- Nếu không đủ bằng chứng, hãy trả lời:
  "Không tìm thấy tài liệu phù hợp."

MỨC ĐỘ TIN CẬY:
{confidence}

CONTEXT:
{context}

CÂU HỎI:
{question}
"""

CONFIDENCE_NOTE = {
    "STRICT": "Kết luận dựa trên quan hệ nghiệp vụ đã xác thực.",
    "RELAXED": "Kết luận dựa trên tài liệu liên quan, chưa xác nhận áp dụng trực tiếp.",
    "ASSISTED": "Kết luận dựa trên xếp hạng ngữ nghĩa.",
    "VECTOR_ONLY": "Kết luận chỉ mang tính tham khảo."
}
MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")

def select_evidence(hits, min_score=0.3, max_docs=5):
    """
    Chọn bằng chứng đủ mạnh để trả lời.
    """
    selected = []
    for h in hits:
        if h.score >= min_score:
            selected.append(h)
        if len(selected) >= max_docs:
            break
    return selected

def build_context(hits, max_chars=3500):
    """
    Đóng gói bằng chứng để LLM không suy diễn.
    """
    blocks = []
    total = 0
    hits = sort_hits_in_order(hits)
    for h in hits:
        text = h.payload.get("chunk_text", "").strip()
        doc_id = h.payload.get("doc_id")
        file_path = h.payload.get("file_url", "N/A")

        block = (
            f"[DOC_ID: {doc_id}]\n"
            f"{text}\n"
            f"NGUỒN: {file_path}\n"
        )

        if total + len(block) > max_chars:
            break

        blocks.append(block)
        total += len(block)

    return "\n---\n".join(blocks)

def can_answer(hits, confidence_level):
    """
    Kiểm soát khi nào được phép trả lời.
    """
    if not hits:
        return False

    if confidence_level == "VECTOR_ONLY":
        # Không cho kết luận nghiệp vụ/pháp lý
        return False

    return True

@lru_cache(maxsize=1)
def _load_qwen():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    device = "cpu"
    if device in ("cuda", "mps"):
        dtype = torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)
    return tokenizer, model, device

def answer_with_qwen_3b(
    tokenizer,
    model,
    device,
    prompt: str,
    max_new_tokens: int = 512
):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # ❗ bắt buộc
            temperature=0.0,        # ❗ chống hallucination
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    return result.strip()

def step5_answer_pipeline(
    question,
    hits,
    confidence_level
):
    tokenizer, model, device = _load_qwen()

    # 1. Validation
    if not hits:
        return "Không tìm thấy tài liệu phù hợp."

    if confidence_level == "VECTOR_ONLY":
        return "Không tìm thấy tài liệu phù hợp."

    # 2. Evidence selection
    evidence = select_evidence(hits)

    # 3. Context packing
    context = build_context(evidence)

    # 4. Build prompt
    prompt = ANSWER_PROMPT.format(
        confidence=CONFIDENCE_NOTE[confidence_level],
        context=context,
        question=question
    )
    # return prompt
    # 5. Generate answer
    return answer_with_qwen_3b(
        tokenizer, model, device, prompt
    )

# =========== Test 10 models LLM ===========
MODELS = [
    # Vietnamese-focused
    "phamhai/Llama-3.2-3B-Instruct-Frog",
    "vilm/vinallama-2.7b-chat",
    "arcee-ai/Arcee-VyLinh",
    "AITeamVN/Vi-Qwen2-3B-RAG",
    "AITeamVN/Vi-Qwen2-1.5B-RAG",
    "ricepaper/vi-gemma-2b-RAG",
    "thangvip/vilord-1.8B-instruct",
    # Multilingual
    "Qwen/Qwen2.5-1.5B-Instruct",
    "facebook/xglm-2.9B",
    "mistralai/Ministral-3-3B-Instruct-2512-BF16",
]

def check_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = check_device()



def build_context_from_payloads(scored_points, top_k=5):
    blocks = []
    for i, sp in enumerate(scored_points[:top_k], start=1):
        p = sp["payload"] if isinstance(sp, dict) else sp.payload
        blocks.append(
            f"[E{i}] doc_no: {p.get('doc_no')}\n"
            f"date: {p.get('date')}\n"
            f"summary: {p.get('summary')}\n"
            f"chunk_index: {p.get('chunk_index')}\n"
            f"file_url: {p.get('file_url')}\n"
            f"text:\n{p.get('chunk_text')}\n"
        )
    return "\n---\n".join(blocks)


def make_prompt(query, context):
    return f"""Bạn là trợ lý QA cho văn bản hành chính tiếng Việt (nguồn OCR).
CHỈ được dùng thông tin trong CONTEXT. Không bịa.
Khi trả lời, phải trích dẫn bằng [E1], [E2]... đúng evidence.

CONTEXT:
{context}

QUESTION:
{query}

YÊU CẦU TRẢ LỜI:
1) Trả lời ngắn gọn, đúng trọng tâm.
2) Nếu hỏi về điều khoản, hãy trích nguyên ý theo OCR và nêu "Điều x".
3) Luôn kèm trích dẫn [E?] sau mỗi ý quan trọng.
4) Nếu CONTEXT không đủ để trả lời, nói rõ "Không đủ dữ liệu trong context".
"""

def load_model(model_id):
    dtype = torch.float16 if DEVICE in ["cuda", "mps"] else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    if DEVICE == "mps":
        model = model.to("mps")
    elif DEVICE == "cpu":
        model = model.to("cpu")

    model.eval()
    return tok, model

def chat_generate(tok, model, prompt, max_new_tokens=256):
    # Lý do: một số model có chat template, dùng sẽ “đúng format” hơn
    if hasattr(tok, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "Bạn là trợ lý QA tiếng Việt."},
            {"role": "user", "content": prompt},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Lý do: do_sample=False giúp giảm lặp/hallucination khi RAG
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.1,
            eos_token_id=tok.eos_token_id,
        )

    decoded = tok.decode(out[0], skip_special_tokens=True)
    return decoded



if __name__=="__main__":
    print("Hybriad Vec Knowgraph Service")

    question = "Các quyết định của sinh viên Tô Thị Ngọc Thiện"
    hits = hybrid_retrieve(
        question,
        driver=driver,
        qdrant=qdrant,
        embedder=model,
        collection=COLLECTION_NAME,
        top_k=5
    )
    print("Retrieved Hits:", hits)
    hits = sort_hits_in_order(hits)
    # test1 = build_context1(hits)
    
    context = build_context_from_payloads(hits)
    QUERY = "Các quyết định của sinh viên Tô Thị Ngọc Thiện"
    prompt = make_prompt(QUERY, context)
    
    
    for mid in MODELS:
        print("\n" + "="*90)
        print("MODEL:", mid)
        try:
            tok, model = load_model(mid)
            ans = chat_generate(tok, model, prompt, max_new_tokens=256)
            print(ans)
        except Exception as e:
            print("ERROR:", repr(e))
        finally:
            # Lý do: giải phóng VRAM/RAM khi chạy nhiều model liên tiếp
            try:
                del tok, model
                torch.cuda.empty_cache()
            except:
                pass
    
    
    # print(f"Retrieved {len(hits)} hits.")
    # result = step5_answer_pipeline(
    #     question=question,
    #     hits=hits,
    #     confidence_level="STRICT"
    # )
    # print("ANSWER:")
    # print(result)

    
