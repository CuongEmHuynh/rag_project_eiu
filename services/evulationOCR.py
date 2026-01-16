

import fitz  # pymupdf
import numpy as np
import cv2
from pathlib import Path
from paddleocr import PaddleOCR
from typing import List, Tuple, Dict, Any, Optional

PDF_PATH = "data/file_minio/27135926-0898-12c8-0a0b-3a100476f984_16-20.pdf"
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)

ocr_det =  PaddleOCR(
        lang="vi",                 
        ocr_version="PP-OCRv5",    

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
        device="cpu",   
        cpu_threads=8,              # "cpu" hoặc "gpu:0" nếu anh build được GPU
    )

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
        cv2.imwrite(str(OUT_DIR / f"page_{i+1:03d}.png"), img)
    return images




import cv2
import numpy as np

def estimate_skew_angle(gray):
    # tìm hướng chữ bằng Hough line trên edge
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
    # radius càng lớn càng "vá" rộng nhưng dễ làm nhòe text gần đó
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

    return bin_img, angle




### ======= Bước 3 Detect và nhận dạng ký tự (OCR) ======== 
def box_bbox(pts: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Mục đích:
      - Chuyển polygon 4 điểm -> bbox (x1,y1,x2,y2) để tính kích thước, lọc, sort.

    Vì sao cần:
      - Lọc nhiễu (box quá nhỏ/quá cao) và sort reading order đều cần bbox.
    """
    xs, ys = pts[:, 0], pts[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def box_wh(pts: np.ndarray) -> Tuple[float, float]:
    """
    Mục đích:
      - Lấy width/height nhanh từ bbox.
    """
    x1, y1, x2, y2 = box_bbox(pts)
    return (x2 - x1), (y2 - y1)

def ensure_poly4(pts: Any) -> np.ndarray:
    """
    Mục đích:
      - Ép mọi dạng polygon output (4 điểm hoặc N điểm) về polygon 4 điểm.

    Vì sao cần:
      - PaddleOCR đôi khi trả polygon nhiều điểm (contour).
      - Bạn muốn downstream crop/warp thống nhất: (4,2) float32.

    Cách làm:
      - Nếu N=4: dùng luôn.
      - Nếu N!=4: dùng minAreaRect để ép về 4 điểm.
    """
    arr = np.array(pts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Invalid polygon shape: {arr.shape}")

    if arr.shape[0] == 4:
        return arr

    rect = cv2.minAreaRect(arr)
    box = cv2.boxPoints(rect)  # (4,2)
    return np.array(box, dtype=np.float32)

def _extract_det_from_predict_output(pred: Any) -> Tuple[List[np.ndarray], List[float]]:
    """
    Mục đích:
      - Chuẩn hoá output từ ocr_det.predict(...) về (boxes, scores)

    Vì sao cần:
      - Output predict() thay đổi theo version: dict hoặc list[dict].
      - Keys hay gặp: dt_polys, det_polys, polys, boxes, dt_scores, scores...

    Output:
      - boxes: list of np.ndarray (4,2) float32
      - scores: list of float (nếu không có score -> 1.0)
    """
    boxes: List[np.ndarray] = []
    scores: List[float] = []

    def handle_one(one: Any):
        nonlocal boxes, scores
        if one is None:
            return

        # case: dict
        if isinstance(one, dict):
            polys = (one.get("dt_polys") or one.get("det_polys") or
                     one.get("polys") or one.get("boxes") or [])
            scs = (one.get("dt_scores") or one.get("det_scores") or
                   one.get("scores") or [])

            for i, p in enumerate(polys):
                try:
                    boxes.append(ensure_poly4(p))
                    if scs and i < len(scs):
                        scores.append(float(scs[i]))
                    else:
                        scores.append(1.0)
                except Exception:
                    continue
            return

        # case: list/tuple (ít gặp trong predict, nhưng vẫn fallback)
        if isinstance(one, (list, tuple)):
            for item in one:
                try:
                    boxes.append(ensure_poly4(item))
                    scores.append(1.0)
                except Exception:
                    continue

    if isinstance(pred, list):
        for one in pred:
            handle_one(one)
    else:
        handle_one(pred)

    # đảm bảo độ dài score
    if len(scores) != len(boxes):
        scores = [1.0] * len(boxes)

    return boxes, scores



def detect_text_boxes(
    img_bgr: np.ndarray,
    ocr_det,
    return_scores: bool = True
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Mục đích:
      - Detect text regions (polygon boxes) trên ảnh BGR/Gray.

    Vì sao cần:
      - Văn bản hành chính cần detect vùng chữ trước để OCR (Step 4) chính xác.
      - PP-OCRv5 detect khá mạnh trên scan tài liệu.

    Input:
      - img_bgr: ảnh BGR hoặc Gray (đã deskew + đã remove red là tốt nhất)
      - ocr_det: PaddleOCR instance (det=True, rec=False)

    Output:
      - boxes: list polygon (4,2) float32
      - scores: độ tự tin box (nếu có)
    """
    if img_bgr is None:
        raise ValueError("img_bgr is None")

    img = np.ascontiguousarray(img_bgr)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    # Ưu tiên predict() (pipeline mới)
    if hasattr(ocr_det, "predict"):
        pred = ocr_det.predict(
            img,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        boxes, scores = _extract_det_from_predict_output(pred)
        return (boxes, scores) if return_scores else (boxes, [])


    raise RuntimeError("ocr_det has neither predict() nor ocr()")


def filter_boxes_basic(
    boxes: List[np.ndarray],
    min_w: int = 25,
    min_h: int = 12,
    max_h: int = 260,
    min_area: int = 300,
    min_aspect: float = 1.1
) -> List[np.ndarray]:
    """
    Mục đích:
      - Loại box nhiễu: quá nhỏ, quá cao, diện tích quá bé.

    Vì sao cần:
      - Scan hay có nhiễu (đốm, nét, viền), detector đôi khi bắt nhầm.
      - Lọc sớm giúp giảm tải Step 4 (recognize) rất nhiều.

    Tham số:
      - min_w/min_h/min_area: giảm -> bắt chữ nhỏ hơn nhưng dễ nhiễu
      - max_h: tránh bắt các khối to bất thường
      - min_aspect: w/h < min_aspect thường là “cục”/nhiễu (tuỳ tài liệu)
    """
    out: List[np.ndarray] = []
    for b in boxes:
        x1, y1, x2, y2 = box_bbox(b)
        w, h = (x2 - x1), (y2 - y1)
        if w < min_w or h < min_h:
            continue
        if h > max_h:
            continue
        if (w * h) < min_area:
            continue
        if h > 0 and (w / h) < min_aspect:
            # với văn bản hành chính chủ yếu là line dài => aspect thường > 2
            # giữ min_aspect thấp để không mất tiêu đề/đoạn ngắn
            pass
        out.append(b)
    return out


def filter_boxes_by_margin(
    boxes: List[np.ndarray],
    img_w: int,
    img_h: int,
    margin: int = 8,
    hard_drop_long_border: bool = True
) -> List[np.ndarray]:
    """
    Mục đích:
      - Giảm trường hợp bắt nhầm viền trang / đường kẻ sát mép.

    Vì sao cần:
      - PDF scan thường có viền/đường biên, detector dễ bắt thành text region.

    Logic:
      - Nếu box sát mép mà quá dài (chiếm 85% chiều rộng/chiều cao) => loại.
      - Tránh loại header sát lề: chỉ hard-drop khi cực dài.
    """
    out: List[np.ndarray] = []
    for b in boxes:
        x1, y1, x2, y2 = box_bbox(b)

        touches = (x1 < margin or y1 < margin or (img_w - x2) < margin or (img_h - y2) < margin)
        if touches and hard_drop_long_border:
            if (x2 - x1) > 0.85 * img_w or (y2 - y1) > 0.85 * img_h:
                continue

        out.append(b)
    return out

def sort_boxes_reading_order(
    boxes: List[np.ndarray],
    line_tol: int = 18
) -> List[np.ndarray]:
    """
    Mục đích:
      - Sắp xếp box theo thứ tự đọc: từ trên xuống, trái qua phải.

    Vì sao cần:
      - OCR ra đúng từng dòng mà xếp sai thứ tự => văn bản lộn xộn, hậu xử lý khó.

    Cách làm:
      - Sort theo y1, sau đó gom “line” theo line_tol
      - Trong cùng line -> sort theo x1
    """
    items = []
    for b in boxes:
        x1, y1, x2, y2 = box_bbox(b)
        items.append((x1, y1, x2, y2, b))

    items.sort(key=lambda t: (t[1], t[0]))

    lines: List[List[tuple]] = []
    for it in items:
        if not lines or abs(it[1] - lines[-1][0][1]) > line_tol:
            lines.append([it])
        else:
            lines[-1].append(it)

    ordered: List[np.ndarray] = []
    for line in lines:
        line.sort(key=lambda t: t[0])
        ordered.extend([t[-1] for t in line])

    return ordered

def draw_boxes(
    img_bgr: np.ndarray,
    boxes: List[np.ndarray],
    save_path: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Mục đích:
      - Vẽ polygon boxes để debug/tune nhanh.
    """
    vis = img_bgr.copy()
    for b in boxes:
        p = b.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [p], isClosed=True, color=color, thickness=thickness)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, vis)
    return vis

def step3_detect_pipeline(
    img_bgr: np.ndarray,
    ocr_det,
    page_id: int = 1,
    debug_dir: Optional[str] = "debug",
    det_score_thr: float = 0.0,  # có thể set 0.3 để lọc box tự tin thấp
    filter_cfg: Optional[Dict[str, Any]] = None,
    line_tol: int = 18
) -> Dict[str, Any]:
    """
    Pipeline Step 3:
      1) Detect raw boxes (PaddleOCR det)
      2) (Optional) lọc theo score
      3) Filter heuristic (size + margin)
      4) Sort reading order
      5) Debug images

    Output:
      - dict gồm raw_boxes, filtered_boxes, ordered_boxes, scores, meta...
    """
    if filter_cfg is None:
        filter_cfg = dict(min_w=25, min_h=12, max_h=260, min_area=300, min_aspect=1.1)

    h, w = img_bgr.shape[:2]

    # 1) detect
    raw_boxes, raw_scores = detect_text_boxes(img_bgr, ocr_det=ocr_det, return_scores=True)

    # 2) filter by score (nếu model trả score)
    if det_score_thr > 0:
        tmp_boxes, tmp_scores = [], []
        for b, s in zip(raw_boxes, raw_scores):
            if s >= det_score_thr:
                tmp_boxes.append(b)
                tmp_scores.append(s)
        raw_boxes, raw_scores = tmp_boxes, tmp_scores

    # 3) heuristic filters
    boxes = filter_boxes_basic(raw_boxes, **filter_cfg)
    boxes = filter_boxes_by_margin(boxes, img_w=w, img_h=h, margin=8)

    # # 4) sort reading order
    ordered = sort_boxes_reading_order(boxes, line_tol=line_tol)

    # 5) debug
    if debug_dir:
        draw_boxes(img_bgr, raw_boxes,     f"{debug_dir}/page_{page_id:03d}_boxes_raw.png")
        draw_boxes(img_bgr, boxes,         f"{debug_dir}/page_{page_id:03d}_boxes_filtered.png")
        draw_boxes(img_bgr, ordered,       f"{debug_dir}/page_{page_id:03d}_boxes_ordered.png")

    return {
        "page_id": page_id,
        "image_size": (h, w),
        "raw_boxes": raw_boxes,
        "raw_scores": raw_scores,
        "filtered_boxes": boxes,
        "ordered_boxes": ordered,
        "config": {
            "det_score_thr": det_score_thr,
            "filter_cfg": filter_cfg,
            "line_tol": line_tol
        }
    }

# ========= main ========
if __name__=="__main__":
    print("This is evulationOCR.py")
    # Render PDF to images
    pages = render_pdf_to_images(PDF_PATH, dpi=350)
    bin_pages = []
    for i, img in enumerate(pages):
        bin_img, angle = preprocess(img)
        bin_pages.append(bin_img)
        cv2.imwrite(str(OUT_DIR / f"page_{i+1:03d}_bin.png"), bin_img)
    
    print("Preprocessing done.")
    print(bin_pages[0].shape)
    # Detect text regions and draw boxes
    # boxs = step3_detect_regions(bin_pages[0], debug_dir="debug", page_id=1)
    out = step3_detect_pipeline(
        img_bgr=bin_pages[0],
    ocr_det=ocr_det,
    page_id=1,
    debug_dir="debug",
    det_score_thr=0.0,  # có thể set 0.3 nếu nhiều nhiễu
    filter_cfg=dict(min_w=25, min_h=12, max_h=260, min_area=250, min_aspect=1.1),
    line_tol=18
)