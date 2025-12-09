
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
import sys

def init_vietocr():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = 'cpu'
    config['cnn']['pretrained'] = True
    config['predictor']['beamsearch'] = True
    return Predictor(config)

def init_PaddleOCR():
    ocr = PaddleOCR(
        # language + version
        lang="vi",                 # hoặc "latin" (vi ∈ nhóm latin trong code nguồn)
        ocr_version="PP-OCRv5",    # "PP-OCRv5" / "PP-OCRv4" / "PP-OCRv3"

        # TẮT các module tiền xử lý để đỡ nặng
        use_doc_orientation_classify=False,  # không xoay toàn trang
        use_doc_unwarping=False,            # không làm phẳng giấy
        use_textline_orientation=False,     # không xoay từng dòng

        # Tham số detection (mapping từ det_* cũ sang text_det_*)
        text_det_limit_side_len=960,  # giống det_limit_side_len
        text_det_limit_type="max",    # giống det_limit_type: "min"/"max"
        text_det_thresh=0.3,          # giống det_db_thresh
        text_det_box_thresh=0.6,      # giống det_db_box_thresh
        text_det_unclip_ratio=1.5,    # giống det_db_unclip_ratio

        # Thiết bị – trên Mac M2 thường dùng CPU
        device="cpu",                 # "cpu" hoặc "gpu:0" nếu anh build được GPU
        # enable_hpi=True,           # nếu muốn bật high performance inference
    )
    return ocr

vietocr = init_vietocr()
paddle_detector = init_PaddleOCR()

# def pdf_to_images(pdf_input, dpi=300, use_bytes=False):
#     """
#     pdf_input: path (str) hoặc bytes (nếu use_bytes=True)
#     return: list[ PIL.Image ]
#     """
#     poppler_path = "/opt/homebrew/bin"
#     if use_bytes:
#         pages = convert_from_bytes(pdf_input, dpi=dpi,poppler_path=poppler_path)
#     else:
#         pages = convert_from_path(pdf_input, dpi=dpi, poppler_path=poppler_path)
#     return pages



def get_text_boxes_and_image(page_pil, detector: PaddleOCR,padding=4):
    """
    Detection từ 1 trang (PIL Image) bằng PaddleOCR.
    Return:
      img_cv: ảnh BGR (numpy)
      boxes: list[(x_min, y_min, x_max, y_max)]
    """
    img_cv = cv2.cvtColor(np.array(page_pil), cv2.COLOR_RGB2BGR)
    results = detector.predict(img_cv,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False)

    if not results:
        return img_cv, []

    res = results[0]

    # Lấy polygon từ detection (ưu tiên dt_polys)
    polys = res.get("dt_polys")
    if polys is None:
        # fallback: dùng rec_polys nếu vì lý do nào đó dt_polys ko có
        polys = res.get("rec_polys")

    if polys is None:
        return img_cv, []

    h, w = img_cv.shape[:2]
    boxes = []

    for poly in polys:
        # poly shape: (4, 2) hoặc (N, 2), mỗi phần tử [x, y]
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]

        x_min = max(0, min(xs) - padding)
        x_max = min(w - 1, max(xs) + padding)
        y_min = max(0, min(ys) - padding)
        y_max = min(h - 1, max(ys) + padding)

        # Lọc bớt box siêu nhỏ (nhiễu)
        if x_max - x_min > 5 and y_max - y_min > 5:
            boxes.append((x_min, y_min, x_max, y_max))

    # Sắp xếp box theo thứ tự đọc: trên xuống, trái sang phải
    boxes.sort(key=lambda b: (b[1], b[0]))

    return img_cv, boxes

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

        #Save image crop for debug
        # pil_crop.save(f"debug_crop_{x_min}_{y_min}_{x_max}_{y_max}.png")

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

    if sys.platform == "win32":
        print("Running on Windows")
        info = pdfinfo_from_path(pdf_path, userpw=None)
        max_pages = info["Pages"]
        for page_number in range(1, max_pages + 1):
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=page_number,
                last_page=page_number,
            )
            # convert_from_path với 1 trang -> images có 1 phần tử
            yield page_number, images[0]
    elif sys.platform == "darwin":
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
        print("Running on macOS")
    else:
        print(f"Running on another OS: {sys.platform}")

    

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
