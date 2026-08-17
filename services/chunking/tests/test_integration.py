"""Integration tests cho feature flag, TXT fallback và tokenizer thật khi cache có sẵn."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from services.chunking.chunk_builder import build_chunks_from_ocr_document
from services.chunking.integration import load_chunks_by_version
from services.chunking.ocr_parser import parse_ocr_data
from services.chunking.tests.fixtures import transfer_fixture
from services.chunking.token_counter import RegexTokenizer, TokenCounter


class _FakeEmbeddingModel:
    """Model stub chỉ cung cấp tokenizer/max length cho fallback test."""

    def __init__(self, max_seq_length: int = 96) -> None:
        """Khởi tạo tokenizer dependency injection."""

        self.tokenizer = RegexTokenizer(max_seq_length)
        self.max_seq_length = max_seq_length


class IntegrationTests(unittest.TestCase):
    """Kiểm tra version routing và tokenizer integration không cần mạng."""

    def test_v2_missing_json_uses_explicit_versioned_txt_fallback(self) -> None:
        """Khi được bật, fallback luôn ghi ``v1-fallback`` và ``OCR_TXT_FALLBACK``."""

        with tempfile.TemporaryDirectory() as directory:
            document_id = "77777777-7777-7777-7777-777777777777"
            Path(directory, f"{document_id}.txt").write_text(
                "Mở đầu\n\nĐiều 1: Nội dung quyết định.", encoding="utf-8"
            )
            chunks = load_chunks_by_version(
                {"Id": document_id, "Summary": "Fallback test"},
                _FakeEmbeddingModel(),
                version="v2",
                ocr_dir=directory,
            )
        children = [chunk for chunk in chunks if chunk.metadata["record_type"] == "child"]
        self.assertTrue(children)
        self.assertTrue(all(chunk.metadata["chunking_version"] == "v1-fallback" for chunk in children))
        self.assertTrue(all(chunk.metadata["source"] == "OCR_TXT_FALLBACK" for chunk in children))

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "transformers chưa được cài")
    def test_cached_real_embedding_tokenizer_has_no_overflow(self) -> None:
        """Integration test dùng PhobertTokenizer thật của model nếu snapshot local tồn tại."""

        snapshot_root = (
            Path.home()
            / ".cache/huggingface/hub/models--bkai-foundation-models--vietnamese-bi-encoder/snapshots"
        )
        snapshots = sorted(
            path.parent
            for path in snapshot_root.glob("*/tokenizer_config.json")
            if (path.parent / "vocab.txt").exists()
        )
        if not snapshots:
            self.skipTest("Không có tokenizer snapshot local; unit tokenizer injection vẫn chạy")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(snapshots[-1]), local_files_only=True)
        counter = TokenCounter(tokenizer=tokenizer, max_seq_length=256)
        chunks = build_chunks_from_ocr_document(
            parse_ocr_data(transfer_fixture("Võ Hoàng Duy")),
            {
                "Id": "88888888-8888-8888-8888-888888888888",
                "Summary": "Quyết định chuyển điểm cho Võ Hoàng Duy",
            },
            counter,
        )
        children = [chunk for chunk in chunks if chunk.metadata["record_type"] == "child"]
        self.assertEqual(tokenizer.__class__.__name__, "PhobertTokenizer")
        self.assertTrue(children)
        self.assertLessEqual(max(chunk.token_count for chunk in children), 256)


if __name__ == "__main__":
    unittest.main()
