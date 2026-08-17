"""Tests boundary và hierarchy của văn bản quyết định."""

from __future__ import annotations

import unittest

from services.chunking.ocr_parser import parse_ocr_data
from services.chunking.structure_parser import build_document_tree, iter_nodes
from services.chunking.tests.fixtures import decision_fixture


class StructureParserTests(unittest.TestCase):
    """Xác nhận Điều/legal basis/recipients không bị gắn sai context."""

    def setUp(self) -> None:
        """Parse fixture một lần cho từng test độc lập."""

        self.document = parse_ocr_data(decision_fixture())
        self.tree = build_document_tree(
            self.document.blocks,
            "11111111-1111-1111-1111-111111111111",
            "Quyết định Hồ Xuân Tường",
        )

    def test_detects_four_articles_without_mid_sentence_split(self) -> None:
        """Substring ``ở điều 1`` trong Điều 2 không tạo boundary mới."""

        articles = list(iter_nodes(self.tree, "article"))
        self.assertEqual([node.title for node in articles], ["Điều 1", "Điều 2", "Điều 3", "Điều 4"])

    def test_legal_basis_is_before_articles(self) -> None:
        """Hai căn cứ nằm trong preamble parent, không nằm trong Điều 1."""

        bases = list(iter_nodes(self.tree, "legal_basis"))
        self.assertEqual(len(bases), 2)
        self.assertTrue(all(self.tree.nodes[node.parent_id].node_type == "preamble" for node in bases))

    def test_recipients_is_not_child_of_last_article(self) -> None:
        """Nơi nhận đóng Article 4 và trở thành root-level section."""

        recipients = list(iter_nodes(self.tree, "recipients"))
        self.assertEqual(len(recipients), 1)
        self.assertEqual(self.tree.nodes[recipients[0].parent_id].node_type, "document")

    def test_noise_blocks_are_not_semantic_nodes(self) -> None:
        """Page number và operator table footnote không tạo node semantic."""

        sources = {source for node in iter_nodes(self.tree) for source in node.source_block_ids}
        self.assertNotIn("page_002_block_0005", sources)
        self.assertNotIn("page_002_block_0008", sources)


if __name__ == "__main__":
    unittest.main()

