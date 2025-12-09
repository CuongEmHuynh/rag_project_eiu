
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from paddleocr import PaddleOCR
from pdf2image import convert_from_path, convert_from_bytes
import cv2
import numpy as np
from PIL import Image
import os
from pdf2image.pdf2image import pdfinfo_from_path
import gc 

def init_vietocr():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = 'cpu'
    config['cnn']['pretrained'] = True
    config['predictor']['beamsearch'] = False
    return Predictor(config)

def init_PaddleOCR():
    return PaddleOCR(use_doc_orientation_classify=True, lang='vi')

vietocr = init_vietocr()
paddle_detector = init_PaddleOCR()

def pdf_to_images(pdf_input, dpi=300, use_bytes=False):
    """
    pdf_input: path (str) hoặc bytes (nếu use_bytes=True)
    return: list[ PIL.Image ]
    """
    poppler_path = "/opt/homebrew/bin"
    if use_bytes:
        pages = convert_from_bytes(pdf_input, dpi=dpi,poppler_path=poppler_path)
    else:
        pages = convert_from_path(pdf_input, dpi=dpi, poppler_path=poppler_path)
    return pages



def get_text_boxes_and_image(page_pil, detector: PaddleOCR):
    """
    Detection từ 1 trang (PIL Image) bằng PaddleOCR.
    Return:
      img_cv: ảnh BGR (numpy)
      boxes: list[(x_min, y_min, x_max, y_max)]
    """
    img_cv = cv2.cvtColor(np.array(page_pil), cv2.COLOR_RGB2BGR)
    result = detector.ocr(img_cv)

    if not result:
        return img_cv, []

    page_result = result[0]  
    boxes = []
    def _poly_to_box(poly):
        arr = np.array(poly)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return None
        x_min = int(np.min(arr[:, 0]))
        y_min = int(np.min(arr[:, 1]))
        x_max = int(np.max(arr[:, 0]))
        y_max = int(np.max(arr[:, 1]))
        return x_min, y_min, x_max, y_max

    if isinstance(page_result, dict):
        polygons = page_result.get("dt_polys") or page_result.get("rec_polys") or []
        for poly in polygons:
            box = _poly_to_box(poly)
            if box:
                boxes.append(box)
    else:
        for line in page_result:
            if not line or not isinstance(line, (list, tuple)) or len(line) < 1:
                continue
            poly = line[0]
            box = _poly_to_box(poly)
            if box:
                boxes.append(box)

    # Sort gần đúng theo dòng đọc: top-to-bottom, left-to-right
    boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))

    return img_cv, boxes_sorted

def recognize_boxes_with_vietocr(img_cv, boxes, recognizer: Predictor):
    """
    img_cv: ảnh BGR (numpy)
    boxes: list[(x_min, y_min, x_max, y_max)]
    return: list[str] text theo thứ tự boxes
    """
    texts = []

    for (x_min, y_min, x_max, y_max) in boxes:
        # Crop vùng text
        crop = img_cv[y_min:y_max, x_min:x_max]  # BGR
        if crop.size == 0:
            continue

        # BGR -> RGB -> PIL
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)

        # Gọi VietOCR
        text = recognizer.predict(pil_crop)
        texts.append(text)

    return texts


def ocr_page_paddle_vietocr(page_pil, detector: PaddleOCR, recognizer: Predictor):
    img_cv, boxes = get_text_boxes_and_image(page_pil, detector)
    if not boxes:
        return ""

    texts = recognize_boxes_with_vietocr(img_cv, boxes, recognizer)

    # Ghép các box thành các dòng đơn giản: mỗi box 1 dòng
    # (Bạn có thể cải tiến grouping theo y để đẹp hơn)
    page_text = "\n".join(texts)
    return page_text

def iter_pdf_pages(pdf_path, dpi=200, poppler_path=None):
    """
    Generator trả về (page_number, PIL.Image) từng trang một.
    Không bao giờ giữ toàn bộ pages trong RAM cùng lúc.
    """
    info = pdfinfo_from_path(pdf_path, userpw=None, poppler_path=poppler_path)
    max_pages = info["Pages"]

    for page_number in range(1, max_pages + 1):
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_number,
            last_page=page_number,
            poppler_path=poppler_path
        )
        # convert_from_path với 1 trang -> images có 1 phần tử
        yield page_number, images[0]

def ocr_pdf_paddle_vietocr(
    pdf_input,
    output_txt="output.txt",
    dpi=200,
    use_bytes=False,
      poppler_path="/opt/homebrew/bin",
    device=None
):
    """
    pdf_input: path str hoặc bytes (nếu use_bytes=True)
    output_txt: đường dẫn file .txt xuất ra
    """
    # 1. Convert PDF -> list ảnh
    # pages = pdf_to_images(pdf_input, dpi=dpi, use_bytes=use_bytes)
    # print(f"Tổng số trang: {len(pages)}")

    # 2. Khởi tạo model
    detector = paddle_detector  # dùng global đã tạo ở trên
    recognizer = vietocr

    all_pages_text = []

    # 3. OCR từng trang
    # for i, page in iter_pdf_pages(pdf_path, dpi=dpi, poppler_path=poppler_path):
    #     print(f"OCR trang {i+1}/{len(pages)} ...")
    #     page_text = ocr_page_paddle_vietocr(page, detector, recognizer)

    #     # Thêm header trang để dễ debug
    #     page_block = f"===== PAGE {i+1} =====\n{page_text}\n"
    #     all_pages_text.append(page_block)

    # # 4. Ghi ra file .txt
    # with open(output_txt, "w", encoding="utf-8") as f:
    #     for block in all_pages_text:
    #         f.write(block)
    #         f.write("\n")
    
    with open(output_txt, "w", encoding="utf-8") as f_out:
        for page_number, page_pil in iter_pdf_pages(pdf_path, dpi=dpi, poppler_path=poppler_path):
            print(f"OCR trang {page_number} ...")

            page_text = ocr_page_paddle_vietocr(
                page_pil,
                detector=paddle_detector,
                recognizer=vietocr
            )

            f_out.write(f"===== PAGE {page_number} =====\n")
            f_out.write(page_text)
            f_out.write("\n\n")

            # Giải phóng biến cục bộ & thu gom rác
            del page_pil, page_text
            gc.collect()


    print(f"Hoàn thành. Đã lưu vào: {os.path.abspath(output_txt)}")

if __name__ == "__main__":
    # Ví dụ sử dụng
    pdf_path= "./data/file_minio/27135926-0898-12c8-0a0b-3a100476f984_16-20.pdf"
    # text = sliding_window_ocr_pdf(pathFile)
    # print(text)
    ocr_pdf_paddle_vietocr(
        pdf_input="./data/file_minio/27135926-0898-12c8-0a0b-3a100476f984_16-20.pdf",
        output_txt="output_paddle_vietocr.txt",
        dpi=200,
        use_bytes=False,
        device=None  
    )
