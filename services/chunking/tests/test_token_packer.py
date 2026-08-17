"""Tests prefix-aware budget và token-window fallback."""

from __future__ import annotations

import unittest

from services.chunking.models import AtomicUnit, ChunkingConfig
from services.chunking.token_counter import RegexTokenizer, TokenCounter
from services.chunking.token_packer import pack_atomic_units


class TokenPackerTests(unittest.TestCase):
    """Không PackedUnit nào được vượt budget sau khi cộng prefix."""

    def test_long_unit_uses_token_window_without_overflow(self) -> None:
        """Long prose không có sentence boundary buộc dùng fallback cuối."""

        counter = TokenCounter(tokenizer=RegexTokenizer(48), max_seq_length=48)
        config = ChunkingConfig(safety_margin_tokens=4, fallback_overlap_tokens=3)
        unit = AtomicUnit(
            unit_id="long",
            unit_type="paragraph",
            parent_id="article-1",
            section_path=["Điều 1"],
            raw_text=" ".join(f"từ{i}" for i in range(100)),
            normalized_text=" ".join(f"từ{i}" for i in range(100)),
            page_start=1,
            page_end=1,
            source_block_ids=["page_001_block_0001"],
            metadata={},
        )

        def builder(units: list[AtomicUnit]) -> str:
            return "Văn bản: Test\nPhần: Điều 1\n\nNội dung:\n" + "\n".join(
                item.normalized_text for item in units
            )

        packed = pack_atomic_units([unit], counter, config, builder)
        self.assertGreater(len(packed), 1)
        for item in packed:
            self.assertLessEqual(counter.count(builder(item.units)), 44)
            self.assertEqual(item.metadata.get("split_fallback"), "token_window")

    def test_does_not_pack_different_section_paths(self) -> None:
        """Hai Khoản cùng type nhưng khác section path vẫn là hai chunks."""

        counter = TokenCounter(tokenizer=RegexTokenizer(128), max_seq_length=128)
        base = dict(
            unit_type="paragraph",
            parent_id="article-1",
            raw_text="Nội dung ngắn",
            normalized_text="Nội dung ngắn",
            page_start=1,
            page_end=1,
            source_block_ids=["page_001_block_0001"],
            metadata={},
        )
        left = AtomicUnit(unit_id="a", section_path=["Điều 1", "Khoản 1"], **base)
        right = AtomicUnit(unit_id="b", section_path=["Điều 1", "Khoản 2"], **base)
        self.assertEqual(len(pack_atomic_units([left, right], counter)), 2)


if __name__ == "__main__":
    unittest.main()

