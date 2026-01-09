from time import sleep
import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, text 
import csv
from pathlib import Path

load_dotenv()

mssql_conn_str = (
    "mssql+pyodbc://{user}:{pwd}@{server}/{db}"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&TrustServerCertificate=yes"
).format(
    user=os.getenv("MSSQL_USER", ""),
    pwd=quote_plus(os.getenv("MSSQL_PASSWORD", "")),
    server=os.getenv("MSSQL_SERVER"),
    db=os.getenv("MSSQL_DB"),
)

mssql_engine = create_engine(mssql_conn_str)

def sql_connection():
    conn = (
        "DRIVER={driver};"
        "SERVER={server};"
        "DATABASE={database};"
        "UID={user};"
        "PWD={pwd};"
        "TrustServerCertificate=yes;"
    ).format(
        driver="{ODBC Driver 17 for SQL Server}",
        server=os.getenv("MSSQL_SERVER",""),
        database=os.getenv("MSSQL_DB",""),
        user=os.getenv("MSSQL_USER","sa"),
        pwd=os.getenv("MSSQL_PASSWORD","Becamex@1234"),
    )
    conn_encoded = quote_plus(conn)
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={conn_encoded}")
    return engine

def fetch_data(mssql_engine):
    query = """
   SELECT TOP 15 D.Id, KeyFileId, [Page], [No], Author, Summary,Column1 as 'DateDocument',RecordId,
    F.Id as 'FileIdMinio', CONCAT(CAST(F.Id AS varchar(36)), '_', ISNULL(F.Name, '')) AS 'FileNameMinio', 'host'+F.[Path] as 'FilePathMinio'
    From Documents D
    LEFT JOIN FileManagementBlobFiles F ON D.KeyFileId=F.MasterKeyId
    where   IsDeleted= 0 and F.CreationTime >='2025-01-03'
    """
    with mssql_engine.connect() as connection:
        result = connection.execute(text(query))
        return result.fetchall()

def _row_to_mapping(row):
    """Normalize Row/dict/tuple data to an ordered mapping."""
    if hasattr(row, "_mapping"):
        return dict(row._mapping)
    if isinstance(row, dict):
        return row
    return {f"col{i}": value for i, value in enumerate(row, start=1)}

def write_rows_to_csv(rows, csv_path="documents_extract.csv"):
    rows = list(rows)
    if not rows:
        print("Không có dữ liệu để xuất ra CSV.")
        return None

    mappings = [_row_to_mapping(row) for row in rows]
    headers = list(mappings[0].keys())

    csv_path = Path(csv_path)
    if csv_path.parent != Path("."):
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(mappings)

    return csv_path

if __name__=="__main__":
    engine = sql_connection()
    data = fetch_data(engine)
    output_file = write_rows_to_csv(data, csv_path="./data/documents.csv")
    if output_file:
        print(f"Đã xuất {len(data)} dòng dữ liệu ra file: {output_file.resolve()}")

 
