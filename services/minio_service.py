import os
import csv
from pathlib import Path
from minio import Minio
from dotenv import load_dotenv

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

def _normalize_object_name(file_path: str, file_name: str) -> str:
    """Build MinIO object path from CSV columns."""
    file_path = (file_path or "").strip()
    file_name = (file_name or "").strip()

    if file_path:
        normalized = file_path.rstrip("/")
        if file_name and not normalized.endswith(file_name):
            normalized = f"{normalized}/{file_name}"
        return normalized.lstrip("/")

    return file_name.lstrip("/").lower()

def download_files_from_csv(
    csv_file: str = "output_documents.csv",
    output_dir: str = "data/file_minio",
    object_column: str = "FilePathMinio",
    file_name_column: str = "FileNameMinio",
):
    csv_path = Path(csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file CSV: {csv_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    minio_client = connect_minio()

    total = success = 0
    with csv_path.open("r", encoding="utf-8", newline="") as csv_handle:
        reader = csv.DictReader(csv_handle)
        for idx, row in enumerate(reader, start=1):
            total += 1
            file_path = row.get(object_column) or row.get(object_column.lower())
            file_name = row.get(file_name_column) or row.get(file_name_column.lower())
            object_name = _normalize_object_name(file_path, file_name.lower())

            if not object_name:
                print(f"[{idx}] Thiếu thông tin đường dẫn MinIO, bỏ qua.")
                continue

            destination_name = file_name or Path(object_name).name or f"file_{idx}"
            destination_file = output_path / destination_name
            destination_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                minio_client.fget_object(
                    minio_bucket,
                    object_name,
                    destination_file.as_posix()
                )
                print(f"[{idx}] Đã tải {object_name} -> {destination_file}")
                success += 1
            except Exception as exc:
                print(f"[{idx}] Lỗi tải {object_name}: {exc}")

    print(f"Hoàn tất: {success}/{total} file được lưu tại {output_path.resolve()}")

if __name__=="__main__":
    csv_source = Path("./data/documents.csv")
    download_files_from_csv(csv_file=str(csv_source), output_dir="data/file_minio")
