from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import fitz  # PyMuPDF
from PIL import Image
import io
import numpy as np


def init_vietocr():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained'] = True
    config['device'] = 'cpu'     # hoặc 'cuda' nếu có GPU
    config['predictor']['beamsearch'] = False

    return Predictor(config)


def init_trocr():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    return processor, model
