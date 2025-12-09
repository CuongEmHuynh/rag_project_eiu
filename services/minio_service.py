
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


def ocr_pdf_file(filepath: str) -> str:
    pdf = fitz.open(filepath)
    full_text = ""

    # Thư mục lưu ảnh từng trang để debug OCR
    abs_path = os.path.abspath(filepath)
    pdf_name = os.path.splitext(os.path.basename(abs_path))[0]
    output_dir = os.path.join(os.path.dirname(abs_path), "ocr_pages", pdf_name)
    os.makedirs(output_dir, exist_ok=True)
    processed_pages_dir = os.path.join(output_dir, "processed_pages")
    windows_root_dir = os.path.join(output_dir, "windows")
    print(f"Lưu ảnh OCR tạm tại: {output_dir}")

    print(f"Số trang trong PDF: {len(pdf)}")
    for page_index in range(len(pdf)):
        page = pdf[page_index]

        # render ảnh DPI cao giúp OCR chính xác hơn
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")

        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Lưu ảnh gốc trước khi đưa qua OCR để dễ kiểm tra lỗi
        image_path = os.path.join(output_dir, f"page_{page_index+1:03d}.png")
        pil_img.save(image_path, format="PNG")
        print(f"Đã lưu trang {page_index+1} tại {image_path}")

        page_label = f"page_{page_index+1:03d}"
        window_debug_dir = os.path.join(windows_root_dir, page_label)

        page_text = ocr_page_vietocr(
            pil_img,
            processed_debug_dir=processed_pages_dir,
            page_label=page_label,
            window_debug_dir=window_debug_dir,
        )
        full_text += f"\n--- PAGE {page_index+1} ---\n{page_text}\n"

    return full_text.strip()    

if __name__=="__main__":
    print("Test Minio Service")
    #pathFile= "./data/file_minio/27135926-0898-12c8-0a0b-3a100476f984_16-20.pdf"
    # minio_client = connect_minio()
    # object_name="host/EIU/HC2017/32632cbf-fd51-25e5-4cce-3a10046ff0ab_1/HC2017.01/27135926-0898-12c8-0a0b-3a100476f984_16-20.pdf"
    # extract_bytes_file(minio_client,object_name)