import fitz  # pymupdf
import numpy as np
import cv2
from pathlib import Path
import requests
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)
#1 Read file pdf, split pages, extract image

#2 Prrocess images, noyise reduction, remove red,...

#3 Call API to extract box sorted by page, extract text from box, save to database

#4 



def estimate_skew_angle(gray):
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=200,
                            minLineLength=gray.shape[1]//3, maxLineGap=20)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:,0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -20 < angle < 20:  # lọc nhiễu
            angles.append(angle)
    return float(np.median(angles)) if angles else 0.0

def deskew(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    angle = estimate_skew_angle(gray)
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle

def build_red_mask(img_bgr, dilate_iter=2):
    """
    Trả về mask vùng đỏ (uint8 0/255).
    Dùng HSV 2 dải vì màu đỏ nằm ở 0..10 và 170..180.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Bạn có thể tune S/V tùy chất lượng scan
    lower1 = np.array([0, 70, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 70, 50])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Làm sạch mask: mở/đóng + dãn nở để bao hết viền dấu
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    return mask

def remove_red_by_inpaint(img_bgr, red_mask):
    """
    Inpaint vùng đỏ để "vá" lại nền.
    Telea thường cho kết quả mượt.
    """
    out = cv2.inpaint(img_bgr, red_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return out

def preprocess(img_bgr):
    img_bgr, angle = deskew(img_bgr)
    
    # Remove red stamps/markss
    red_mask = build_red_mask(img_bgr, dilate_iter=2)
    img_bgr = remove_red_by_inpaint(img_bgr, red_mask)

    # denoise nhẹ để giữ nét dấu tiếng Việt
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # adaptive threshold phù hợp nền scan không đều
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15
    )

    # chữ thường là đen trên nền trắng, đảm bảo đúng chiều
    # nếu nền đen chữ trắng -> đảo
    if np.mean(bin_img) < 127:
        bin_img = 255 - bin_img

    return bin_img, angle, img_bgr

def render_pdf_to_images(pdf_path: str, dpi: int = 300):
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    images = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    return images

if __name__=="__main__":
    url_extract_layout = "http://localhost:8000/layout/detect"
    
    # Thay tất cac đường dẫn PDF ở đây để test
    pdf_path = "data/file_minio/27135926-0898-12c8-0a0b-3a100476f984_16-20.pdf"
   
    pages = render_pdf_to_images(pdf_path)
    for i, img in enumerate(pages):
        bin_img, angle, img_bgr = preprocess(img)
        ok,file = cv2.imencode('.png', img_bgr)
        r = requests.post(url_extract_layout, files={ "file": (f"page_{i+1}.png", file.tobytes(), "image/png")})
        print(f"Page {i+1} - Skew angle: {angle:.2f} degrees - API response: {r.status_code} {r.text}")
            