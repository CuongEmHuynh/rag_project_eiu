
from time import sleep
import os
from PIL import Image
from minio import Minio
from dotenv import load_dotenv
import io
load_dotenv()

# Config Minio
minio_bucket = os.getenv("MINIO_BUCKET","trung-tam-luu-tru")

def connect_minio():
    """
    Connect to Minio server and return Minio client.
    """
    minio_client =Minio(
        endpoint=os.getenv("MINIO_ENDPOINT","localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY","minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY","minioadmin"),
        secure=False)
    return minio_client


def extract_bytes_file(minio_client: Minio, object_name: str) -> str:
    """
    Get file bytes from Minio by object name.
    
    :param minio_client: Description
    :type minio_client: Minio
    :param object_name: Description
    :type object_name: str
    :return: Description
    :rtype: str
    """
    response = minio_client.get_object(minio_bucket, object_name)
    try:
        data = response.read()
        print(f"Đọc {len(data)} byte từ {object_name}")
        return data
    finally:
        response.close()
        response.release_conn()

if __name__=="__main__":
    print("Test Minio Service")
    #pathFile= "./data/file_minio/27135926-0898-12c8-0a0b-3a100476f984_16-20.pdf"
    # minio_client = connect_minio()
    # object_name="host/EIU/HC2017/32632cbf-fd51-25e5-4cce-3a10046ff0ab_1/HC2017.01/27135926-0898-12c8-0a0b-3a100476f984_16-20.pdf"
    # extract_bytes_file(minio_client,object_name)