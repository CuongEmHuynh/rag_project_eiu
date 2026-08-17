"""End-to-end tests từ OCR dict đến parent/child/payload."""

from __future__ import annotations

import unittest

from services.chunking.chunk_builder import build_chunks_from_ocr_document
from services.chunking.ocr_parser import parse_ocr_data
from services.chunking.retrieval_text import chunk_to_payload
from services.chunking.tests.fixtures import decision_fixture, transfer_fixture
from services.chunking.token_counter import RegexTokenizer, TokenCounter


class ChunkBuilderTests(unittest.TestCase):
    """Kiểm tra output v2 deterministic, token-safe và có đầy đủ provenance."""

    def setUp(self) -> None:
        """Dùng tokenizer fake injection theo đúng allowance của spec unit test."""

        self.counter = TokenCounter(tokenizer=RegexTokenizer(160), max_seq_length=160)

    def _build(self, data: dict, document_id: str, summary: str):
        """Helper chạy pipeline với metadata tối thiểu nhưng thực tế."""

        return build_chunks_from_ocr_document(
            parse_ocr_data(data),
            {
                "Id": document_id,
                "Summary": summary,
                "No": "01/QĐ-EIU",
                "Author": "Trường Đại học Quốc tế Miền Đông",
                "DateDocument": "01/01/2016",
            },
            self.counter,
        )

    def test_decision_children_have_existing_parent_and_no_overflow(self) -> None:
        """Mọi child của quyết định 4 Điều có parent và retrieval token-safe."""

        chunks = self._build(
            decision_fixture(),
            "11111111-1111-1111-1111-111111111111",
            "Quyết định Hồ Xuân Tường",
        )
        parents = {chunk.chunk_id for chunk in chunks if chunk.metadata["record_type"] == "parent"}
        children = [chunk for chunk in chunks if chunk.metadata["record_type"] == "child"]
        self.assertTrue(children)
        self.assertTrue(all(chunk.parent_id in parents for chunk in children))
        self.assertTrue(all(chunk.token_count <= self.counter.max_seq_length for chunk in children))
        self.assertFalse(any("None" in chunk.retrieval_text for chunk in children))

    def test_cross_page_table_rows_are_children_of_article(self) -> None:
        """PHYS row page 2 giữ Article parent, table parent và schema source/target."""

        chunks = self._build(
            transfer_fixture("Võ Hoàng Duy"),
            "22222222-2222-2222-2222-222222222222",
            "Quyết định chuyển điểm cho sinh viên Võ Hoàng Duy",
        )
        table_rows = [chunk for chunk in chunks if chunk.chunk_type == "table_row"]
        self.assertGreaterEqual(len(table_rows), 4)
        phys = next(chunk for chunk in table_rows if "PHYS 201" in chunk.retrieval_text)
        self.assertEqual(phys.page_start, 2)
        self.assertTrue(phys.metadata.get("table_parent_id"))
        self.assertTrue(phys.metadata.get("cross_page_table"))
        self.assertIn("Môn học được chuyển", phys.retrieval_text)
        self.assertIn("Điều 1", phys.section_path)

    def test_ids_are_deterministic(self) -> None:
        """Cùng OCR/meta tạo cùng ordered list UUID5."""

        args = (
            decision_fixture(),
            "33333333-3333-3333-3333-333333333333",
            "Quyết định deterministic",
        )
        left = self._build(*args)
        right = self._build(*args)
        self.assertEqual([chunk.chunk_id for chunk in left], [chunk.chunk_id for chunk in right])

    def test_payload_contains_required_v2_fields(self) -> None:
        """Payload storage có section/page/raw/retrieval/token/version."""

        meta = {"Id": "44444444-4444-4444-4444-444444444444", "Summary": "Test"}
        chunks = build_chunks_from_ocr_document(
            parse_ocr_data(decision_fixture()), meta, self.counter
        )
        child = next(chunk for chunk in chunks if chunk.metadata["record_type"] == "child")
        payload = chunk_to_payload(child, meta)
        for key in (
            "section_path", "chunk_type", "page_start", "page_end", "raw_text",
            "retrieval_text", "token_count", "chunking_version",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["chunking_version"], "v2")

    def test_three_acceptance_documents_build(self) -> None:
        """Cả quyết định và hai student transfer fixtures đều sinh child chunks."""

        fixtures = [
            (decision_fixture(), "55555555-5555-5555-5555-555555555551", "Hồ Xuân Tường"),
            (transfer_fixture("Phạm Minh Quân"), "55555555-5555-5555-5555-555555555552", "Phạm Minh Quân"),
            (transfer_fixture("Võ Hoàng Duy"), "55555555-5555-5555-5555-555555555553", "Võ Hoàng Duy"),
        ]
        for data, document_id, summary in fixtures:
            with self.subTest(summary=summary):
                chunks = self._build(data, document_id, summary)
                self.assertTrue(any(chunk.metadata["record_type"] == "child" for chunk in chunks))


if __name__ == "__main__":
    unittest.main()

