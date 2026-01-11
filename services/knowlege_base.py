import pandas as pd 
from neo4j import GraphDatabase

# setting cho kết nối Neo4j
NEO4J_URI = "bolt://222.255.214.30:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "StrongPassword123!"


driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

def write_full_graph(tx, d):
    tx.run("""
    // Organization
    MERGE (o:Organization {id:$org_id})
    SET o.name = $org_name

    // DocumentBlock
    MERGE (db:DocumentBlock {code:$db_code})
    SET db.name = $db_name
    MERGE (o)-[:MANAGES]->(db)

    // TableOfContent
    MERGE (t:TableOfContent {code:$toc_code})
    SET t.name = $toc_name
    MERGE (db)-[:HAS_TOC]->(t)

    // Box
    MERGE (b:Box {code:$box_code})
    MERGE (t)-[:CONTAINS_BOX]->(b)

    // RecordType
    MERGE (rt:RecordType {code:$record_type_code})
    SET rt.name = $record_type_name

    // Record
    MERGE (r:Record {id:$record_id})
    SET r.code = $record_code,
        r.name = $record_name
    MERGE (b)-[:CONTAINS_RECORD]->(r)
    MERGE (r)-[:HAS_TYPE]->(rt)

    // Document
    MERGE (d:Document {id:$doc_id})
    SET d.no = $doc_no,
        d.page = $doc_page,
        d.date = $doc_date,
        d.summary = $doc_summary
    MERGE (r)-[:HAS_DOCUMENT]->(d)

    // FileObject
    MERGE (f:FileObject {id:$file_id})
    SET f.name = $file_name,
        f.path = $file_path
    MERGE (d)-[:STORED_AS]->(f)
    """, **d)


def write_document(tx, doc):
    tx.run("""
    MERGE (d:Document {id: $doc_id})
    SET d.no = $no,
        d.page = $page,
        d.date = $date

    MERGE (r:Record {id: $record_id})
    SET r.code = $record_code,
        r.name = $record_name,
        r.type = $record_type
    MERGE (d)-[:BELONGS_TO]->(r)

    MERGE (f:FileObject {id: $file_id})
    SET f.path = $file_path,
        f.name = $file_name
    MERGE (d)-[:STORED_AS]->(f)

    MERGE (o:Organization {name: $author})
    MERGE (d)-[:ISSUED_BY]->(o)
    """, {
        "doc_id": doc["doc_id"],
        "no": doc["document_meta"]["no"],
        "page": doc["document_meta"]["page"],
        "date": doc["document_meta"]["date"],

        "record_id": doc["record_meta"]["record_id"],
        "record_code": doc["record_meta"]["record_code"],
        "record_name": doc["record_meta"]["record_name"],
        "record_type": doc["record_meta"]["record_type"],

        "file_id": doc["file_meta"]["file_id"],
        "file_path": doc["file_meta"]["file_path"],
        "file_name": doc["file_meta"]["file_name"],

        "author": doc["org_meta"]["author"]
    })



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
    
    print("Inserting documents into Neo4j...")
    with driver.session() as session:
        for doc in documents:
            session.execute_write(write_full_graph, doc)
    
    print("Done.")

if __name__ == "__main__":
    main()