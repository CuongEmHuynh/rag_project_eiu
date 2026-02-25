import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ---------------------------
# Geometry helpers
# ---------------------------
def clip_bbox(b, w, h):
    x0, y0, x1, y1 = b
    x0, x1 = sorted([int(round(x0)), int(round(x1))])
    y0, y1 = sorted([int(round(y0)), int(round(y1))])
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w - 1))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))
    return [x0, y0, x1, y1]

def area(b):
    x0, y0, x1, y1 = b
    return max(0, x1 - x0) * max(0, y1 - y0)

def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = area(a) + area(b) - inter
    return 0.0 if ua == 0 else inter / ua

def contains(outer, inner, ratio=0.95):
    # inner nằm trong outer theo tỉ lệ diện tích giao/diện tích inner
    ix = max(0, min(outer[2], inner[2]) - max(outer[0], inner[0]))
    iy = max(0, min(outer[3], inner[3]) - max(outer[1], inner[1]))
    inter = ix * iy
    return area(inner) > 0 and (inter / area(inner)) >= ratio

# ---------------------------
# Union-Find for clustering
# ---------------------------
class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

# ---------------------------
# Label policy
# ---------------------------
PRIORITY = {
    "table": 0,
    "title": 1,
    "table_caption": 2,
    "figure_caption": 3,
    "plain text": 4,
    "figure": 5,
    "abandon": 9,
}

OCR_FLAG = {
    "table": True,          # hoặc False nếu bạn chỉ muốn detect table và xử lý khác
    "title": True,
    "plain text": True,
    "table_caption": True,
    "figure_caption": True,
    "figure": False,        # thường không OCR chữ ký/mộc trong luồng chính
    "abandon": False,
}

def pick_best(blocks):
    # blocks: list dict có category_name, score
    return sorted(
        blocks,
        key=lambda b: (PRIORITY.get(b["category_name"], 99), -b.get("score", 0.0))
    )[0]

# ---------------------------
# Main standardization
# ---------------------------
def standardize_page_blocks(raw_blocks: List[Dict], w: int, h: int,
                            iou_dup=0.90, min_area=200,
                            drop_text_inside_table=True) -> List[Dict]:

    # 1) normalize bbox + basic filter
    blocks = []
    for b in raw_blocks:
        label = b["category_name"]
        bbox = clip_bbox(b["bbox_xyxy"], w, h)
        if area(bbox) < min_area:
            continue
        blocks.append({
            "label_raw": label,
            "score": float(b.get("score", 1.0)),
            "bbox": bbox,
            "reading_order": int(b.get("reading_order", 10**9)),
        })

    # 2) drop abandon early (nhưng vẫn xử lý dedup ở bước 3 nếu bạn muốn giữ lại)
    blocks_no_abandon = [b for b in blocks if b["label_raw"] != "abandon"]

    # 3) deduplicate clusters by IoU
    n = len(blocks_no_abandon)
    dsu = DSU(n)
    for i in range(n):
        for j in range(i + 1, n):
            if iou(blocks_no_abandon[i]["bbox"], blocks_no_abandon[j]["bbox"]) >= iou_dup:
                dsu.union(i, j)

    clusters = {}
    for i in range(n):
        r = dsu.find(i)
        clusters.setdefault(r, []).append(i)

    deduped = []
    for _, idxs in clusters.items():
        cand = []
        for k in idxs:
            b = blocks_no_abandon[k]
            cand.append({
                "category_name": b["label_raw"],
                "score": b["score"],
                "bbox": b["bbox"],
                "reading_order": b["reading_order"],
            })
        best = pick_best(cand)
        deduped.append(best)

    # 4) optional: drop text/title/caption inside table
    if drop_text_inside_table:
        tables = [b for b in deduped if b["category_name"] == "table"]
        if tables:
            kept = []
            for b in deduped:
                if b["category_name"] in ("plain text", "title", "table_caption", "figure_caption"):
                    inside_any = any(contains(t["bbox"], b["bbox"], ratio=0.90) for t in tables)
                    if inside_any:
                        continue
                kept.append(b)
            deduped = kept

    # 5) finalize schema + order
    # sort by original reading_order first (đúng với output của bạn), fallback by y0,x0
    deduped.sort(key=lambda b: (b.get("reading_order", 10**9), b["bbox"][1], b["bbox"][0]))

    out = []
    for i, b in enumerate(deduped, start=1):
        out.append({
            "id": i,
            "label": b["category_name"].upper().replace(" ", "_"),
            "score": float(b.get("score", 1.0)),
            "bbox": b["bbox"],
            "order": i,
            "ocr": bool(OCR_FLAG.get(b["category_name"], True)),
        })
    return out

def standardize_layout_json(layout_json_path: str, page_sizes: Dict[str, Tuple[int,int]], out_path="layout_blocks.standard.json"):
    with open(layout_json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    out_pages = []
    for p in doc["pages"]:
        pid = str(p["page_id"])
        if pid not in page_sizes:
            raise ValueError(f"Missing page size for page_id={pid}")
        w, h = page_sizes[pid]
        std_blocks = standardize_page_blocks(p["blocks"], w, h)
        out_pages.append({"page_id": pid, "width": w, "height": h, "blocks": std_blocks})

    out_doc = {"input": doc.get("input"), "task": doc.get("task"), "num_pages": doc.get("num_pages"), "pages": out_pages}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False, indent=2)
    return out_path