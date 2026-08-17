"""OCR JSON fixtures tổng hợp theo ba case acceptance trong đặc tả."""

from __future__ import annotations

from typing import Any


COURSE_HEADER = """
<thead>
  <tr>
    <th colspan="4">Môn học SV đã học</th>
    <th colspan="4">Môn học SV được chuyển</th>
  </tr>
  <tr>
    <th>Mã MH</th><th>Tên môn học</th><th>Số TC</th><th>Điểm</th>
    <th>Mã MH chuyển</th><th>Tên môn học</th><th>Số TC được chuyển</th><th>Điểm chuyển đổi</th>
  </tr>
</thead>
"""


def course_table(rows: list[list[str]], include_header: bool = True) -> str:
    """Tạo HTML table 8 cột; continuation có thể cố ý đặt data row trong thead."""

    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    if include_header:
        return f"<table>{COURSE_HEADER}<tbody>{body}</tbody></table>"
    return f"<table><thead>{body}</thead></table>"


def decision_fixture() -> dict[str, Any]:
    """Case quyết định 4 Điều, legal bases, recipients và signature."""

    return {
        "input_file": "quyet_dinh_ho_xuan_tuong.pdf",
        "page_count": 2,
        "pages": [
            {
                "page_number": 1,
                "blocks": [
                    {"type": "abandon", "bbox": [0.05, 0.03, 0.45, 0.08], "content": "TRƯỜNG ĐẠI HỌC QUỐC TẾ MIỀN ĐÔNG"},
                    {"type": "title", "bbox": [0.35, 0.10, 0.65, 0.15], "content": "QUYẾT ĐỊNH"},
                    {"type": "text", "bbox": [0.10, 0.20, 0.90, 0.25], "content": "Căn cứ Luật Giáo dục;"},
                    {"type": "text", "bbox": [0.10, 0.26, 0.90, 0.31], "content": "- Căn cử Quy chế đào tạo;"},
                    {"type": "text", "bbox": [0.10, 0.34, 0.90, 0.40], "content": "Điều 1: Công nhận kết quả cho sinh viên Hồ Xuân Tường."},
                    {"type": "text", "bbox": [0.10, 0.43, 0.90, 0.50], "content": "1. Phòng Đào tạo cập nhật kết quả."},
                    {"type": "text", "bbox": [0.10, 0.54, 0.90, 0.61], "content": "Điều 2: Sinh viên có tên ở điều 1 phải hoàn tất học phí."},
                ],
            },
            {
                "page_number": 2,
                "blocks": [
                    {"type": "text", "bbox": [0.10, 0.10, 0.90, 0.17], "content": "Điều 3. Các đơn vị liên quan chịu trách nhiệm thi hành."},
                    {"type": "text", "bbox": [0.10, 0.20, 0.90, 0.27], "content": "Điều 4: Quyết định có hiệu lực kể từ ngày ký."},
                    {"type": "text", "bbox": [0.10, 0.60, 0.40, 0.67], "content": "Nơi nhận:"},
                    {"type": "text", "bbox": [0.10, 0.68, 0.50, 0.74], "content": "- Như Điều 4;"},
                    {"type": "table_footnote", "bbox": [0.10, 0.77, 0.30, 0.80], "content": "Quỳnh (ĐT)"},
                    {"type": "text", "bbox": [0.60, 0.65, 0.90, 0.72], "content": "HIỆU TRƯỞNG"},
                    {"type": "figure_caption", "bbox": [0.60, 0.82, 0.90, 0.88], "content": "Nguyễn Văn A"},
                    {"type": "abandon", "bbox": [0.45, 0.96, 0.55, 0.99], "content": "53"},
                ],
            },
        ],
    }


def transfer_fixture(student_name: str, include_math: bool = True) -> dict[str, Any]:
    """Case course-transfer table bắt đầu trang 1 và tiếp tục không header ở trang 2."""

    page_one_rows = [
        ["ENG 101", "Tiếng Anh 1", "3", "B+", "ENG 111", "Anh văn cơ bản", "3", "B+"],
    ]
    if include_math:
        page_one_rows.append(
            ["MATH 151", "Toán ứng dụng 1", "4", "A-", "MATH 101", "Giải tích 1A", "4", "A"]
        )
    page_two_rows = [
        ["PHYS 201", "Vật lý 1A", "4", "B", "PHYS 101", "Vật lý đại cương", "4", "B+"],
        ["CSE 101", "Nhập môn lập trình", "3", "A", "CSE 110", "Kỹ thuật lập trình", "3", "A"],
    ]
    return {
        "input_file": f"quyet_dinh_{student_name.lower().replace(' ', '_')}.pdf",
        "page_count": 2,
        "pages": [
            {
                "page_number": 1,
                "blocks": [
                    {"type": "title", "bbox": [0.35, 0.06, 0.65, 0.12], "content": "QUYẾT ĐỊNH"},
                    {"type": "text", "bbox": [0.10, 0.17, 0.90, 0.23], "content": f"Điều 1: Chuyển điểm cho sinh viên {student_name} như bảng sau:"},
                    {"type": "table", "bbox": [0.05, 0.28, 0.95, 0.95], "content": course_table(page_one_rows)},
                ],
            },
            {
                "page_number": 2,
                "blocks": [
                    {"type": "table", "bbox": [0.05, 0.04, 0.95, 0.55], "content": course_table(page_two_rows, include_header=False)},
                    {"type": "text", "bbox": [0.10, 0.60, 0.90, 0.67], "content": "Kết quả chuyển điểm được cập nhật vào hệ thống."},
                    {"type": "text", "bbox": [0.10, 0.72, 0.90, 0.79], "content": "Điều 2: Các đơn vị liên quan chịu trách nhiệm thi hành."},
                    {"type": "text", "bbox": [0.10, 0.84, 0.40, 0.89], "content": "Nơi nhận:"},
                ],
            },
        ],
    }

