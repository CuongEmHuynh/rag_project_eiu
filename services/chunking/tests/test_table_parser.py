"""Tests HTML schema flattening, course semantics và cross-page reconstruction."""

from __future__ import annotations

import unittest

from services.chunking.models import ChunkingConfig
from services.chunking.ocr_parser import group_blocks_by_page, parse_ocr_data
from services.chunking.structure_parser import build_document_tree
from services.chunking.table_parser import (
    parse_document_tables,
    parse_html_table,
    reconstruct_cross_page_tables,
    serialize_table_row,
)
from services.chunking.tests.fixtures import course_table, transfer_fixture


class TableParserTests(unittest.TestCase):
    """Kiểm tra table row luôn giữ schema và context nguồn/đích."""

    def test_flattens_multilevel_header(self) -> None:
        """Group header colspan được nối với leaf header tương ứng."""

        table = parse_html_table(
            course_table([["MATH 151", "Toán ứng dụng 1", "4", "A-", "MATH 101", "Giải tích 1A", "4", "A"]])
        )
        self.assertEqual(table.column_count, 8)
        self.assertIn("Môn học SV đã học", table.headers[0])
        self.assertIn("Mã MH", table.headers[0])
        self.assertEqual(table.metadata["schema_type"], "course_transfer")

    def test_course_row_serialization_has_source_and_target_labels(self) -> None:
        """Retrieval row chứa đủ 4 giá trị chính và hai nhóm semantic."""

        row = ["MATH 151", "Toán ứng dụng 1", "4", "A-", "MATH 101", "Giải tích 1A", "4", "A"]
        table = parse_html_table(course_table([row]))
        text = serialize_table_row(
            table,
            row,
            {"Summary": "Quyết định chuyển điểm Võ Hoàng Duy"},
            ["QUYẾT ĐỊNH", "Điều 1", "Bảng chuyển điểm"],
        )
        for value in ("MATH 151", "Toán ứng dụng 1", "MATH 101", "Giải tích 1A"):
            self.assertIn(value, text)
        self.assertIn("Môn học đã học", text)
        self.assertIn("Môn học được chuyển", text)

    def test_reconstructs_cross_page_table_and_inherits_schema(self) -> None:
        """Page 2 data-in-thead được ghép vào table page 1 và dùng schema cũ."""

        document = parse_ocr_data(transfer_fixture("Võ Hoàng Duy"))
        tree = build_document_tree(document.blocks, "22222222-2222-2222-2222-222222222222")
        physical = parse_document_tables(document.blocks, tree)
        logical = reconstruct_cross_page_tables(
            physical,
            group_blocks_by_page(document.blocks),
            threshold=ChunkingConfig().table_continuation_threshold,
        )
        self.assertEqual(len(physical), 2)
        self.assertEqual(len(logical), 1)
        self.assertEqual((logical[0].page_start, logical[0].page_end), (1, 2))
        self.assertTrue(logical[0].metadata["cross_page"])
        serialized = serialize_table_row(
            logical[0], logical[0].rows[-2], {}, ["Điều 1", "Bảng"], include_context=False
        )
        self.assertIn("PHYS 201", serialized)
        self.assertIn("PHYS 101", serialized)
        self.assertIn("Môn học đã học", serialized)


if __name__ == "__main__":
    unittest.main()

