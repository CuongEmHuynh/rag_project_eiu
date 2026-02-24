import json
import re
import unicodedata

try:
    import pandas as pd
except ImportError:
    pd = None

EASYOCR_PATH  = "out/result_OCR/page_001_Esayocr_results.json"
PADDLEOCR_PATH = "out/result_OCR/page_001_Paddleocr_results.json"
VIETOCR_PATH  = "out/result_OCR/page_001_Vietocr_results.json"

GROUND_TRUTH_PATH = "out/result_OCR/page_001_Ground.json"
# Ví dụ nếu có GT thật:
# GROUND_TRUTH_PATH = "/mnt/data/page_001_ground_truth.json"

# 05) Hàm đọc JSON list[{id, text, ...}] -> dict: id -> text
def load_ocr_json(path: str) -> dict:
    # 05.1 Mở file với encoding utf-8 để giữ đúng tiếng Việt có dấu
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    # 05.2 Tạo dict theo id để align dòng nhanh (O(1) lookup)
    by_id = {}
    for it in items:
        by_id[it["id"]] = it.get("text", "")
    return by_id

# 06) Chuẩn hoá text để giảm “nhiễu” khi so sánh (unicode/space/quotes)
def normalize_text(s: str) -> str:
    # 06.1 Nếu None -> chuỗi rỗng để tránh lỗi
    if s is None:
        return ""

    # 06.2 Chuẩn hoá unicode dạng NFKC để đồng nhất ký tự (rất quan trọng với tiếng Việt)
    s = unicodedata.normalize("NFKC", s)

    # 06.3 Đồng nhất các dạng dấu nháy thông minh thành nháy thường
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')

    # 06.4 Gom nhiều whitespace thành 1 khoảng trắng và trim hai đầu
    s = re.sub(r"\s+", " ", s).strip()

    return s

# 07) Tokenizer cho WER (tách theo “từ”)
def tokenize_words(s: str) -> list:
    # 07.1 Normalize + đưa về lowercase để giảm sai khác do hoa/thường
    s = normalize_text(s).lower()

    # 07.2 Đổi dấu câu thành khoảng trắng (giữ chữ cái có dấu tiếng Việt)
    #     \w: chữ/số/_ ; thêm dải À-ỹ để giữ tiếng Việt
    s = re.sub(r"[^\wÀ-ỹ']+", " ", s, flags=re.UNICODE)

    # 07.3 Gom whitespace và trim
    s = re.sub(r"\s+", " ", s).strip()

    # 07.4 Nếu rỗng -> []
    if not s:
        return []
    return s.split(" ")

# 08) Levenshtein edit distance (dùng chung cho char-list hoặc word-list)
def levenshtein(a, b) -> int:
    # 08.1 Độ dài 2 chuỗi (hoặc list)
    n, m = len(a), len(b)

    # 08.2 Trường hợp biên: một bên rỗng
    if n == 0:
        return m
    if m == 0:
        return n

    # 08.3 Tối ưu RAM: đảm bảo m <= n để DP chỉ lưu 1 hàng có kích thước nhỏ hơn
    if m > n:
        a, b = b, a
        n, m = m, n

    # 08.4 prev là hàng DP trước (0..m)
    prev = list(range(m + 1))

    # 08.5 Duyệt từng phần tử của a (i từ 1..n)
    for i in range(1, n + 1):
        # 08.5.1 cur[0] = i (phải xoá i ký tự để về rỗng)
        cur = [i] + [0] * m
        ai = a[i - 1]

        # 08.5.2 Duyệt từng phần tử của b (j từ 1..m)
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1  # thay thế nếu khác
            # 08.5.3 DP transition: insert / delete / replace
            cur[j] = min(
                prev[j] + 1,        # delete
                cur[j - 1] + 1,     # insert
                prev[j - 1] + cost  # replace
            )

        # 08.5.4 Cập nhật hàng trước
        prev = cur

    # 08.6 Kết quả là ô cuối
    return prev[m]

# 09) Tính CER theo chuẩn micro-average: tổng distance / tổng độ dài GT
def compute_cer_micro(gt_by_id: dict, hyp_by_id: dict, ids: list) -> float:
    total_dist = 0
    total_len = 0
    for i in ids:
        gt = normalize_text(gt_by_id.get(i, ""))
        hyp = normalize_text(hyp_by_id.get(i, ""))

        total_dist += levenshtein(list(gt), list(hyp))
        total_len += len(gt)

 
    return (total_dist / total_len) if total_len else 0.0

# 10) Tính WER theo chuẩn micro-average: tổng distance(word) / tổng số từ GT
def compute_wer_micro(gt_by_id: dict, hyp_by_id: dict, ids: list) -> float:
    total_dist = 0
    total_words = 0
    for i in ids:
        gt_w = tokenize_words(gt_by_id.get(i, ""))
        hyp_w = tokenize_words(hyp_by_id.get(i, ""))

        total_dist += levenshtein(gt_w, hyp_w)
        total_words += len(gt_w)

    return (total_dist / total_words) if total_words else 0.0

# 11) Line Accuracy: % dòng khớp hoàn toàn sau normalize
def compute_line_accuracy(gt_by_id: dict, hyp_by_id: dict, ids: list) -> float:
    correct = 0
    for i in ids:
        if normalize_text(gt_by_id.get(i, "")) == normalize_text(hyp_by_id.get(i, "")):
            correct += 1
    return correct / len(ids) if ids else 0.0

# 12) Load 3 mô hình OCR
easy_by_id   = load_ocr_json(EASYOCR_PATH)
paddle_by_id = load_ocr_json(PADDLEOCR_PATH)
viet_by_id   = load_ocr_json(VIETOCR_PATH)

# 13) Load GT:
#     - Nếu có GROUND_TRUTH_PATH: dùng GT thật
#     - Nếu chưa có: demo dùng VietOCR làm proxy GT
if GROUND_TRUTH_PATH:
    gt_by_id = load_ocr_json(GROUND_TRUTH_PATH)
    gt_name = "ground_truth"
else:
    gt_by_id = viet_by_id
    gt_name = "vietocr_proxy_gt"

# 14) Danh sách id để đánh giá (giao của GT và các mô hình)
ids = sorted(set(gt_by_id.keys()) & set(easy_by_id.keys()) & set(paddle_by_id.keys()) & set(viet_by_id.keys()))

# 15) Tính metrics cho từng model
models = {
    "easyocr": easy_by_id,
    "paddleocr": paddle_by_id,
    "vietocr": viet_by_id,
}

rows = []
for name, hyp in models.items():
    cer = compute_cer_micro(gt_by_id, hyp, ids)
    wer = compute_wer_micro(gt_by_id, hyp, ids)
    la  = compute_line_accuracy(gt_by_id, hyp, ids)

    rows.append({
        "model": name,
        "GT": gt_name,
        "CER": cer,
        "WER": wer,
        "LineAccuracy": la,
        "n_lines": len(ids),
    })

# 16) In bảng tổng hợp
if pd:
    df = pd.DataFrame(rows)
    df["CER"] = (df["CER"] * 100).map(lambda x: f"{x:.2f}%")
    df["WER"] = (df["WER"] * 100).map(lambda x: f"{x:.2f}%")
    df["LineAccuracy"] = (df["LineAccuracy"] * 100).map(lambda x: f"{x:.2f}%")
    print(df.sort_values(["CER", "WER"]))
else:
    # 16.1 Nếu không có pandas thì in dạng text đơn giản
    for r in rows:
        print(r)
