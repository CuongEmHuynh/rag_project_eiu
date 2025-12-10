import pandas as pd
from paddleocr import PPStructureV3
from paddlex import create_pipeline
import cv2
from io import StringIO

def init_vietocr_table():
    table_engine  = create_pipeline(
        pipeline="PP-StructureV3",
        params={
            # OCR tiếng Việt
            "lang": "vi",
            "ocr_version": "PP-OCRv5",

            # Tắt các bước preprocess nặng mà PDF scan phẳng không cần
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,

            # Giới hạn kích thước ảnh khi detect text
            "text_det_limit_side_len": 1200,
            "text_det_limit_type": "max",
            "text_det_thresh": 0.3,
            "text_det_box_thresh": 0.6,
            "text_det_unclip_ratio": 1.5,
            "text_rec_score_thresh": 0.5,

            # BẬT nhận dạng bảng & (tuỳ chọn) con dấu
            "use_table_recognition": True,
            "use_seal_recognition": True,

            # Không dùng công thức / chart / region detection cho nhẹ
            "use_formula_recognition": False,
            "use_chart_recognition": False,
            "use_region_detection": False,
        },
    )
    return table_engine



def image_to_tables(img_path: str, pipeline: PPStructureV3):
    """
    Trả về list DataFrame cho tất cả bảng trong ảnh.
    """
    img = cv2.imread(img_path)  # BGR
    result_iter = pipeline.predict(
        input=img_path,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_table_recognition=True,
        use_seal_recognition=False,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_region_detection=True,
    )
      # hoặc pipeline(img_path)

    tables = []

    for res in result_iter:
        # res là một object – convert sang dict cho dễ xử lý
        if hasattr(res, "to_dict"):
            data = res.to_dict()
        else:
            data = getattr(res, "res", {})

        layout = data.get("layout", {})
        blocks = layout.get("blocks", [])

        for block in blocks:
            if block.get("type") != "table":
                continue

            table_res = block.get("res", {}) or {}
            # tuỳ phiên bản: có thể là "html" hoặc "structure_html"
            html = (
                table_res.get("html")
                or table_res.get("structure_html")
                or ""
            )

            if not html:
                continue

            try:
                dfs = pd.read_html(StringIO(html))
            except ValueError:
                # html không parse được
                continue

            if not dfs:
                continue

            tables.append(dfs[0])
    return tables

if __name__ == "__main__":
    vietocr_table_engine = init_vietocr_table()
    path_file = "./data/file_minio/table.png"
    tables = image_to_tables(path_file, vietocr_table_engine)
    for i, table in enumerate(tables):
        print(f"Table {i}:")
        print(table)
    print("Viet OCR Table ")