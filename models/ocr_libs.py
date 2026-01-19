# services/ocr_libs.py

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import cv2
import numpy as np

def init_vietocr(device="cpu"):
    cfg = Cfg.load_config_from_name("vgg_transformer")  # hoặc "transformer"
    cfg["device"] = device
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