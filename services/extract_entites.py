import os
import json
import re
from functools import lru_cache

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from neo4j import GraphDatabase

SYSTEM_PROMPT = """
Bạn là hệ thống trích xuất thông tin.
Nhiệm vụ: trích xuất thực thể ngữ nghĩa từ văn bản hành chính.

Quy tắc:
- KHÔNG trả lời câu hỏi.
- KHÔNG tóm tắt.
- KHÔNG tự ý đoán thông tin thiếu.
- Chỉ trích xuất Person và Event.
- Trả về DUY NHẤT JSON hợp lệ.
"""

USER_PROMPT = """
Hãy trích xuất tri thức ngữ nghĩa từ đoạn văn bản dưới đây.

Định nghĩa:
- Person: người thật được nhắc trong văn bản.
- Event: hành động hành chính (quyết định, kỷ luật, khen thưởng).

Loại Event hợp lệ:
- DECISION
- DISCIPLINARY_ACTION
- REWARD
- OTHER

Quan hệ hợp lệ:
- MENTIONS (Document → Person)
- DESCRIBES (Document → Event)
- APPLIES_TO (Event → Person)

Trả về JSON theo schema:
{{
  "persons": [{{ "id": "...", "name": "..." }}],
  "events": [{{ "id": "...", "type": "...", "description": "..." }}],
  "relations": [{{ "source": "...", "predicate": "...", "target": "..." }}]
}}

Văn bản:
\"\"\"
{TEXT}
\"\"\"
"""
MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "512"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
NEO4J_URI = "bolt://222.255.214.30:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "StrongPassword123!"


driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)




#  -----Use your OpenAI API key ---
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,api_key="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")

# llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0)
# def extract_semantic_kg(text: str) -> dict:
#     print("Extracting semantic KG...")
    
#     """
#     Purpose:
#     - Convert raw OCR text into structured semantic knowledge.

#     Why:
#     - Text itself is NOT knowledge.
#     - Knowledge must be structured before storing in Graph.
#     """
#     messages = [
#         SystemMessage(content=SYSTEM_PROMPT),
#         HumanMessage(content=USER_PROMPT.format(TEXT=text))
#     ]
#     response = llm.invoke(messages)

#     return json.loads(response.content)


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


def extract_semantic_kg_Qwen(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    tokenizer, model, device = _load_qwen()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(TEXT=text)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.0,
        )
    out_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    print("Model output:", out_text)
    print("Type of out_text:", type(out_text))
    # Nếu model có thêm text, cắt JSON ra trước khi json.loads
    match = re.search(r"\{.*\}", out_text, re.S)
    if not match:
        raise ValueError("Model output does not contain JSON")
    json_text = match.group(0).strip()
    return json.loads(json_text)

def validate_kg(kg: dict):
    """
    Purpose:
    - Prevent malformed or hallucinated data from entering Graph.
    """
    if not isinstance(kg, dict):
        raise ValueError("KG must be a dict")

    for key in ["persons", "events", "relations"]:
        if key not in kg:
            raise ValueError(f"Missing key: {key}")

    for p in kg["persons"]:
        if "id" not in p or "name" not in p:
            raise ValueError("Invalid Person entity")

    for e in kg["events"]:
        if "id" not in e or "type" not in e or "description" not in e:
            raise ValueError("Invalid Event entity")

def normalize_id(prefix: str, value: str) -> str:
    """
    Purpose:
    - Create stable, reproducible IDs for Graph nodes.
    """
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return f"{prefix}_{safe}"


def write_semantic_graph(tx, doc_id: str, kg: dict):
    """
    Purpose:
    - Persist semantic knowledge into Neo4j.
    """
    # Persons
    for p in kg["persons"]:
        pid = normalize_id("person", p["name"])

        tx.run("""
        MERGE (p:Person {id:$pid})
        SET p.name = $name
        """, pid=pid, name=p["name"])

        tx.run("""
        MATCH (d:Document {id:$doc_id}), (p:Person {id:$pid})
        MERGE (d)-[:MENTIONS]->(p)
        """, doc_id=doc_id, pid=pid)

    # Events
    for e in kg["events"]:
        eid = normalize_id("event", e["description"])

        tx.run("""
        MERGE (e:Event {id:$eid})
        SET e.type = $type,
            e.description = $desc
        """, eid=eid, type=e["type"], desc=e["description"])

        tx.run("""
        MATCH (d:Document {id:$doc_id}), (e:Event {id:$eid})
        MERGE (d)-[:DESCRIBES]->(e)
        """, doc_id=doc_id, eid=eid)

    # Event → Person
    for r in kg["relations"]:
        if r["predicate"] == "APPLIES_TO":
            tx.run("""
            MATCH (e:Event {id:$eid}), (p:Person {id:$pid})
            MERGE (e)-[:APPLIES_TO]->(p)
            """, eid=r["source"], pid=r["target"])



def build_document_objects(df: pd.DataFrame):
    """Chuyển đổi DataFrame thành danh sách các đối tượng tài liệu."""
    docs = []
    for _, r in df.iterrows():
        docs.append({
            # Document
            "doc_id": r["Id"],
            "doc_no": r["No"],
            "doc_page": r["Page"],
            "doc_date": r["DateDocument"],
            "doc_summary": r["Summary"],

            # File
            "file_id": r["FileIdMinio"],
            "file_name": r["FileNameMinio"],
            "file_path": r["FilePathMinio"],

            # Record
            "record_id": r["RecordId"],
            "record_code": r["RecordCode"],
            "record_name": r["RecordName"],

            # RecordType
            "record_type_code": r["RecordTypeCode"],
            "record_type_name": r["RecordTypeName"],

            # Box
            "box_code": r["BoxCode"],

            # TOC
            "toc_code": r["TableOfContentCode"],
            "toc_name": r["TableOfContentName"],

            # DocumentBlock
            "db_code": r["DocumentBlockCode"],
            "db_name": r["DocumentBlockName"],

            # Organization
            "org_id": r.get("OrganizationId", r["Author"]),
            "org_name": r["Author"]
        })
    return docs

def main():
    
    CSV_PATH= "data/query_data1.csv"
    df = pd.read_csv(CSV_PATH)
    
    print("Building document objects...")
    documents = build_document_objects(df)
    with driver.session() as session:
        for doc in documents:
            kg = extract_semantic_kg_Qwen(doc["doc_summary"])
            validate_kg(kg)
            print(f"Document ID: {doc['doc_id']}")
            session.execute_write(
            write_semantic_graph,
            doc["doc_id"],
            kg)
    
    print("Inserting documents extract done.")    

if __name__ == "__main__":
    main()

    
