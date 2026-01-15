import os
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from sentence_transformers import SentenceTransformer

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
COLLECTION_NAME="rag_document_v1"
MODEL_EMBEDDING="bkai-foundation-models/vietnamese-bi-encoder"
COLLECTION_NAME="rag_document_v1"
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
        return []   # fallback → Vector-first


# ===== Qdrant Vector Search =====
def vector_search_filtered(qdrant, collection, query_vector, allowed_doc_ids, limit=5):
    print(allowed_doc_ids)
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
        allowed_doc_ids= [doc_id.upper() for doc_id in allowed_doc_ids]
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


def build_context(hits):
    blocks = []
    for h in hits:
        blocks.append(
            f"- (doc_id={h.payload['doc_id']}) {h.payload['chunk_text']}\n"
            f"  File: {h.payload.get('file_path','')}"
        )
    return "\n".join(blocks)


if __name__=="__main__":
    print("Hybriad Vec Knowgraph Service")

    question = "Các quyết định của sinh viên Lương Quân Vương"
    hits = hybrid_retrieve(
        question,
        driver=driver,
        qdrant=qdrant,
        embedder=model,
        collection=COLLECTION_NAME,
        top_k=5
    )

    context = build_context(hits)
    print(context)

    
