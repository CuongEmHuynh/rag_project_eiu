# services/ocr_libs.py

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import cv2
import numpy as np

from paddleocr import PaddleOCR


import easyocr
import pytesseract
import re
import unicodedata


#==== VietOCR ====#
def init_vietocr(device="cpu"):
    cfg = Cfg.load_config_from_name("vgg_transformer")  # hoặc "transformer"
    cfg["device"] = device
    # cfg["cnn"]["pretrained"] = False
    cfg["predictor"]["beamsearch"] = True
    return Predictor(cfg)

vietocr = init_vietocr(device="cpu")

def vietocr_recognize(gray_or_bgr: np.ndarray) -> str:
    if gray_or_bgr.ndim == 2:
        rgb = cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return vietocr.predict(pil)


#==== PaddleOCR ====#
ocr_rec = PaddleOCR(
    lang="vi",                 
    ocr_version="PP-OCRv5",    
    device="cpu",   
    cpu_threads=8, 
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,      # Không detect
)

def paddle_recognize(crop_bgr: np.ndarray):
    res = ocr_rec.predict(crop_bgr)
    if not res or not res[0]:
        return "", 0.0
    # Khi chỉ recognition, res là list các tuple (text, score)
    one = res[0] if isinstance(res, list) else res
    text = one["rec_texts"][0]
    score = float(one["rec_scores"][0])
    return text, score


easy_reader = easyocr.Reader(["vi", "en"], gpu=False)

def easyocr_recognize(crop_bgr: np.ndarray):
    out = easy_reader.readtext(crop_bgr)
    if not out:
        return "", 0.0
    out.sort(key=lambda x: x[2], reverse=True)
    return out[0][1], float(out[0][2])

def tesseract_recognize(gray_or_bgr: np.ndarray) -> str:
    if gray_or_bgr.ndim == 3:
        gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bgr
    config = r'--oem 1 --psm 6 -l vie'
    return pytesseract.image_to_string(gray, config=config).strip()