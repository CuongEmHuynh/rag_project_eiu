
from time import sleep
import os
import uuid
from io import BytesIO
from typing import Callable, Optional
from PIL import Image
from minio import Minio
from dotenv import load_dotenv
import fitz  # PyMuPDF
import io
# from paddleocr import PaddleOCR


load_dotenv()

# Config Minio
minio_bucket = os.getenv("MINIO_BUCKET","trung-tam-luu-tru")
BASE_TXT_PATH ="data_txt"
OCR_LANG = os.getenv("TESSERACT_LANG", "vie+eng")
OCR_CONFIG = os.getenv("TESSERACT_CONFIG", "--psm 6")
POPPLER_PATH = os.getenv("POPPLER_PATH")
OCR_DPI = int(os.getenv("PDF_OCR_DPI", "300"))
VIETOCR_CONFIG_NAME = os.getenv("VIETOCR_CONFIG_NAME", "vgg_transformer")
VIETOCR_DEVICE = os.getenv("VIETOCR_DEVICE", "cpu")
VIETOCR_WEIGHTS_PATH = os.getenv("VIETOCR_WEIGHTS_PATH","./config/vgg_transformer.pth")
VIETOCR_BEAMSEARCH = os.getenv("VIETOCR_BEAMSEARCH", "false")
VIETOCR_MIN_RESULT_LENGTH = int(os.getenv("VIETOCR_MIN_RESULT_LENGTH", "5"))
VIETOCR_FALLBACK_WINDOW_WIDTH = int(os.getenv("VIETOCR_FALLBACK_WINDOW_WIDTH", "700"))
VIETOCR_FALLBACK_WINDOW_HEIGHT = int(os.getenv("VIETOCR_FALLBACK_WINDOW_HEIGHT", "180"))
VIETOCR_FALLBACK_STRIDE_X = int(os.getenv("VIETOCR_FALLBACK_STRIDE_X", "350"))
VIETOCR_FALLBACK_STRIDE_Y = int(os.getenv("VIETOCR_FALLBACK_STRIDE_Y", "120"))


def _str_to_bool(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")

# config OCR model (Viet OCR)

def init_vietocr():
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg

    config = Cfg.load_config_from_name(VIETOCR_CONFIG_NAME)
    config['cnn']['pretrained'] = True
    config['device'] = VIETOCR_DEVICE
    config['predictor']['beamsearch'] = _str_to_bool(VIETOCR_BEAMSEARCH)

    if VIETOCR_WEIGHTS_PATH:
        if os.path.isfile(VIETOCR_WEIGHTS_PATH):
            config['weights'] = VIETOCR_WEIGHTS_PATH
        else:
            raise FileNotFoundError(
                f"Không tìm thấy file weights VietOCR tại {VIETOCR_WEIGHTS_PATH}. "
                "Kiểm tra lại biến môi trường VIETOCR_WEIGHTS_PATH."
            )

    return Predictor(config)

def init_trocr():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    return processor, model

vietocr = init_vietocr()
# trocr_processor, trocr_model = init_trocr()
## End Config OCRs


#config 
# ocr = PaddleOCR(
#     lang="vi",                    
#     use_doc_orientation_classify=False, 
#     use_doc_unwarping=False,           
#     use_textline_orientation=False      
# )


def ocr_page_vietocr(
    pil_img: Image.Image,
    processed_debug_dir: Optional[str] = None,
    page_label: Optional[str] = None,
    window_debug_dir: Optional[str] = None,
):
    processed = preprocess_image(
        pil_img,
        debug_dir=processed_debug_dir,
        debug_label=page_label,
    )
    text = vietocr.predict(processed)

    normalized = text.strip()
    if normalized and normalized != "1" and len(normalized) >= VIETOCR_MIN_RESULT_LENGTH:
        return text

    print("[VietOCR] Kết quả quá ngắn, sử dụng fallback sliding-window.")
    fallback_text = _sliding_window_vietocr_page(pil_img, debug_dir=window_debug_dir)
    return fallback_text if fallback_text.strip() else text


def preprocess_image(
    pil_img: Image.Image,
    debug_dir: Optional[str] = None,
    debug_label: Optional[str] = None,
):
    import cv2
    import numpy as np

    img = np.array(pil_img)

    # Bảo đảm ảnh ở dạng 3 kênh RGB
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 1. Khử màu đỏ của con dấu bằng mask HSV + inpaint
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)
    clean = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)

    # 2. Grayscale + giảm nhiễu nhẹ
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. Tăng tương phản cục bộ bằng adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        10
    )

    # 4. Làm sạch nét chữ
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel, iterations=1)

    processed_img = Image.fromarray(opened).convert("RGB")

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        filename = f"{debug_label}_processed.png" if debug_label else f"processed_{uuid.uuid4().hex}.png"
        processed_img.save(os.path.join(debug_dir, filename))

    return processed_img


def _preprocess_window_for_vietocr(crop: "np.ndarray"):
    import cv2
    import numpy as np

    if crop.ndim == 2:
        rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    elif crop.shape[2] == 4:
        rgb = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
    else:
        rgb = crop

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.medianBlur(thresh, 3)

    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)


def _sliding_window_vietocr_page(
    pil_img: Image.Image,
    debug_dir: Optional[str] = None,
) -> str:
    import numpy as np

    win_w = VIETOCR_FALLBACK_WINDOW_WIDTH
    win_h = VIETOCR_FALLBACK_WINDOW_HEIGHT
    stride_x = max(1, VIETOCR_FALLBACK_STRIDE_X)
    stride_y = max(1, VIETOCR_FALLBACK_STRIDE_Y)

    np_img = np.array(pil_img.convert("RGB"))
    h, w, _ = np_img.shape
    results = []

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    for y in range(0, h, stride_y):
        for x in range(0, w, stride_x):
            crop = np_img[y:min(y + win_h, h), x:min(x + win_w, w)]
            if crop.size == 0:
                continue

            processed = _preprocess_window_for_vietocr(crop)
            processed_pil = Image.fromarray(processed)

            if debug_dir:
                window_name = f"win_y{y:04d}_x{x:04d}.png"
                processed_pil.save(os.path.join(debug_dir, window_name))

            try:
                text = vietocr.predict(processed_pil)
            except Exception as exc:
                print(f"[VietOCR] Lỗi khi đọc cửa sổ ({x}, {y}): {exc}")
                continue

            cleaned = text.strip()
            if cleaned:
                results.append((y, x, cleaned))

    results.sort(key=lambda item: (item[0], item[1]))
    return "\n".join(item[2] for item in results)


def connect_minio():
    minio_client =Minio(
        endpoint=os.getenv("MINIO_ENDPOINT","localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY","minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY","minioadmin"),
        secure=False)
    return minio_client

def _extract_pdf_text(pdf_bytes: bytes) -> str:
    def _attempt(fn: Callable[[bytes], Optional[str]]) -> str:
        try:
            extracted = fn(pdf_bytes)
        except Exception as exc:  # pragma: no cover - helper diagnostics
            print(f"Extractor {fn.__name__} lỗi: {exc}")
            return ""

        if extracted and extracted.strip():
            return extracted
        return ""

    def _use_pymupdf(data: bytes) -> Optional[str]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return None

        doc = fitz.open(stream=data, filetype="pdf")
        try:
            return "\n".join(page.get_text("text") for page in doc)
        finally:
            doc.close()

    def _use_pdfplumber(data: bytes) -> Optional[str]:
        try:
            import pdfplumber
        except ImportError:
            return None

        with pdfplumber.open(BytesIO(data)) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)

    def _use_pdfminer(data: bytes) -> Optional[str]:
        try:
            from pdfminer.high_level import extract_text
        except ImportError:
            return None

        return extract_text(BytesIO(data))

    def _use_pypdf(data: bytes) -> Optional[str]:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            return None

        reader = PdfReader(BytesIO(data), strict=False)
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    # def _use_ocr_paddle(data: bytes) -> Optional[str]:
    #     from typing import Optional
    #     import io
    #     import fitz  # PyMuPDF
    #     import numpy as np
    #     from PIL import Image
    #     try:
    #         pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    #     except Exception as e:
    #         print(f"[OCR] Không mở được PDF: {e}")
    #         return None

    #     full_text = ""
    #     for page_index in range(len(pdf)):
    #         page = pdf[page_index]

    #         # Render trang PDF -> ảnh PNG dạng bytes
    #         pix = page.get_pixmap()
    #         img_bytes = pix.tobytes("png")

    #         img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    #         img_np = np.array(img)

    #         # ⭐ 3.x: dùng predict, KHÔNG dùng cls=
    #         result_list = ocr.predict(img_np)

    #         page_text = ""

    #         # result_list là list các Result object
    #         for res in result_list:
    #             # Mỗi res.json là 1 dict, trong đó 'res' chứa 'rec_texts'
    #             data = res.json.get("res", {})
    #             texts = data.get("rec_texts", [])
    #             # Ghép tất cả dòng text lại
    #             page_text += " ".join(texts) + " "

    #         full_text += f"\n--- PAGE {page_index + 1} ---\n{page_text}"

    #         return full_text.strip() if full_text else None

    # def _use_ocr(data: bytes) -> Optional[str]:
    #     try:
    #         from pdf2image import convert_from_bytes
    #         import pytesseract
    #     except ImportError:
    #         return None

    #     convert_kwargs = {"dpi": OCR_DPI, "fmt": "png"}
    #     if POPPLER_PATH:
    #         convert_kwargs["poppler_path"] = POPPLER_PATH

    #     images = convert_from_bytes(data, **convert_kwargs)
    #     ocr_text = []
    #     for image in images:
    #         gray = image.convert("L")
    #         ocr_text.append(
    #             pytesseract.image_to_string(
    #                 gray,
    #                 lang=OCR_LANG,
    #                 config=OCR_CONFIG,
    #             ).strip()
    #         )
    #     return "\n".join(filter(None, ocr_text))


    def ocr_pdf_vietocr(pdf_bytes: bytes) -> str:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        final_text = ""

        for idx in range(len(pdf)):
            page = pdf[idx]
            pix = page.get_pixmap(dpi=200)  # DPI cao sẽ OCR tốt hơn
            image_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            page_text = ocr_page_vietocr(img)
            final_text += f"\n--- PAGE {idx+1} ---\n{page_text}\n"

        return final_text.strip()

    for extractor in (_use_pymupdf, _use_pdfplumber, _use_pdfminer, _use_pypdf,ocr_pdf_vietocr):
        text = _attempt(extractor)
        if text:
            return text

    raise RuntimeError(
        "Không đọc được nội dung PDF. Đảm bảo file không bị mã hóa và cài thêm pymupdf/pdfplumber/pdfminer/PyPDF2 hoặc OCR."
    )


def extract_text_file(minio_client: Minio, object_name: str) -> str:
    response = minio_client.get_object(minio_bucket, object_name)
    try:
        data = response.read()
        print(f"Đọc {len(data)} byte từ {object_name}")
        text = _extract_pdf_text(data)
        print(f"Trích xuất {len(text)} ký tự từ PDF")
        return text
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
    pathFile= "./data/file_minio/27135926-0898-12c8-0a0b-3a100476f984_16-20.pdf"
    # minio_client = connect_minio()
    # object_name="host/EIU/HC2017/32632cbf-fd51-25e5-4cce-3a10046ff0ab_1/HC2017.01/27135926-0898-12c8-0a0b-3a100476f984_16-20.pdf"
    
    result =  ocr_pdf_file(pathFile)
    
    # extract_text_file(minio_client,object_name)
