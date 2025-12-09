from time import sleep
import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, text 

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
    SELECT D.Id, KeyFileId, [Page], [No], Author, Summary,Column1 as 'DateDocument',RecordId,
    F.Id as 'FileIdMinio', CONCAT(CAST(F.Id AS varchar(36)), '_', ISNULL(F.Name, '')) AS 'FileNameMinio', 'host'+F.[Path] as 'FilePathMinio'
    From Documents D
    LEFT JOIN FileManagementBlobFiles F ON D.KeyFileId=F.MasterKeyId
    where   IsDeleted= 0
    """
    with mssql_engine.connect() as connection:
        result = connection.execute(text(query))
        return result.fetchall()
    
if __name__=="__main__":
    engine = sql_connection()
    data = fetch_data(engine)
    for row in data:
        sleep(5)
        print(row)
 
