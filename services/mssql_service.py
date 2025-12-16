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
    where   IsDeleted= 0
    """
    with mssql_engine.connect() as connection:
        result = connection.execute(text(query))
        return result.fetchall()
    
def write_row_to_csv(row, csv_path="output_row.csv"):
    p = Path(csv_path)
    is_new = not p.exists()

    # hỗ trợ sqlalchemy Row object (has _mapping), dict, hoặc tuple/list
    if hasattr(row, "_mapping"):
        mapping = dict(row._mapping)
        headers = list(mapping.keys())
        values = [mapping[k] for k in headers]
    elif isinstance(row, dict):
        headers = list(row.keys())
        values = [row[k] for k in headers]
    else:
        # tuple/list -> tạo header col1, col2...
        values = list(row)
        headers = [f"col{i}" for i in range(1, len(values)+1)]

    mode = "w" if is_new else "a"
    with p.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(headers)
        writer.writerow(values)    



if __name__=="__main__":
    engine = sql_connection()
    data = fetch_data(engine)
    print(data[0])

 
