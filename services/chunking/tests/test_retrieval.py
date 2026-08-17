"""Tests per-parent deduplication và adaptive context expansion."""

from __future__ import annotations

import unittest

from services.chunking.retrieval import deduplicate_children, expand_context


class RetrievalTests(unittest.TestCase):
    """Child search không bị một parent chiếm toàn top-k và expansion đúng loại."""

    def test_deduplicates_per_parent(self) -> None:
        """Giới hạn hai hits cùng parent nhưng giữ ranking order."""

        hits = [
            {"score": 1 - index / 10, "payload": {"chunk_id": str(index), "parent_id": "p1"}}
            for index in range(4)
        ]
        hits.append({"score": 0.5, "payload": {"chunk_id": "x", "parent_id": "p2"}})
        output = deduplicate_children(hits, max_children_per_parent=2)
        self.assertEqual([item["payload"]["chunk_id"] for item in output], ["0", "1", "x"])

    def test_adaptive_table_row_uses_table_parent(self) -> None:
        """Exact table hit lấy schema/table context thay vì full Article mặc định."""

        hit = {
            "score": 0.9,
            "payload": {
                "chunk_id": "c1",
                "chunk_type": "table_row",
                "parent_id": "article-1",
                "table_parent_id": "table-1",
            },
        }
        output = expand_context(
            [hit],
            {"article-1": {"chunk_id": "article-1"}, "table-1": {"chunk_id": "table-1"}},
        )
        self.assertEqual(output[0].context[0]["chunk_id"], "table-1")


if __name__ == "__main__":
    unittest.main()
