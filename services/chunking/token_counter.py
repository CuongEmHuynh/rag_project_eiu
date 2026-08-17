"""Adapter tokenizer thật của embedding model và tokenizer nhẹ cho test/debug offline."""

from __future__ import annotations

import re
from typing import Any


class TokenizerResolutionError(RuntimeError):
    """Không thể tìm tokenizer/max sequence length đáng tin cậy từ model."""


class TokenCounter:
    """Đếm token bằng tokenizer gắn với SentenceTransformer/embedding model.

    Có thể dependency-inject ``tokenizer`` và ``max_seq_length`` cho unit test.
    Production nên truyền chính ``embedding_model`` để tránh lệch tokenizer.
    """

    def __init__(
        self,
        model: Any | None = None,
        *,
        tokenizer: Any | None = None,
        max_seq_length: int | None = None,
    ) -> None:
        """Resolve tokenizer và giới hạn model theo public attribute trước, fallback defensive."""

        self.model = model
        self.tokenizer = tokenizer or self._resolve_tokenizer(model)
        self.max_seq_length = max_seq_length or self._resolve_max_seq_length(model, self.tokenizer)
        if self.max_seq_length <= 0:
            raise TokenizerResolutionError("max_seq_length phải lớn hơn 0")

    def count(self, text: str) -> int:
        """Đếm token đầy đủ, bao gồm special tokens khi tokenizer hỗ trợ."""

        return len(self._encode(text, add_special_tokens=True))

    def encode_content(self, text: str) -> list[int]:
        """Encode body không special tokens để phục vụ token-window fallback."""

        return self._encode(text, add_special_tokens=False)

    def decode_content(self, token_ids: list[int]) -> str:
        """Decode token window về text mà không tự thêm special tokens."""

        decode = getattr(self.tokenizer, "decode", None)
        if not callable(decode):
            raise TokenizerResolutionError("Tokenizer không có decode(), không thể token-window split")
        try:
            return str(decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)).strip()
        except TypeError:
            return str(decode(token_ids)).strip()

    def _encode(self, text: str, add_special_tokens: bool) -> list[int]:
        """Gọi tokenizer qua ``encode`` hoặc callable API, luôn tắt truncation."""

        encode = getattr(self.tokenizer, "encode", None)
        if callable(encode):
            try:
                value = encode(
                    text or "",
                    add_special_tokens=add_special_tokens,
                    truncation=False,
                )
            except TypeError:
                value = encode(text or "", add_special_tokens=add_special_tokens)
            return _flatten_token_ids(value)

        if callable(self.tokenizer):
            value = self.tokenizer(
                text or "",
                add_special_tokens=add_special_tokens,
                truncation=False,
            )
            if isinstance(value, dict):
                value = value.get("input_ids", [])
            else:
                value = getattr(value, "input_ids", [])
            return _flatten_token_ids(value)
        raise TokenizerResolutionError("Tokenizer không callable và không có encode()")

    @staticmethod
    def _resolve_tokenizer(model: Any | None) -> Any:
        """Ưu tiên ``model.tokenizer``, sau đó fallback SentenceTransformer module đầu."""

        if model is None:
            raise TokenizerResolutionError("Cần embedding model hoặc tokenizer explicit")
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        first_module = getattr(model, "_first_module", None)
        if callable(first_module):
            tokenizer = getattr(first_module(), "tokenizer", None)
            if tokenizer is not None:
                return tokenizer
        try:
            tokenizer = getattr(model[0], "tokenizer", None)
        except (KeyError, IndexError, TypeError, AttributeError):
            tokenizer = None
        if tokenizer is None:
            raise TokenizerResolutionError("Không resolve được tokenizer từ embedding model")
        return tokenizer

    @staticmethod
    def _resolve_max_seq_length(model: Any | None, tokenizer: Any) -> int:
        """Lấy model max length; bỏ qua sentinel cực lớn của HuggingFace tokenizer."""

        candidates = [
            getattr(model, "max_seq_length", None) if model is not None else None,
            getattr(tokenizer, "model_max_length", None),
        ]
        for value in candidates:
            try:
                parsed = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if 0 < parsed < 1_000_000:
                return parsed
        raise TokenizerResolutionError("Model/tokenizer không công bố max_seq_length hợp lệ")


class RegexTokenizer:
    """Tokenizer deterministic nhẹ chỉ dành cho unit test/debug offline explicit.

    Nó không thay thế tokenizer embedding trong production. Mỗi word/punctuation là
    một token; hai special tokens được thêm khi yêu cầu.
    """

    model_max_length: int

    def __init__(self, model_max_length: int = 256) -> None:
        """Khởi tạo vocabulary động và max length do caller chỉ định."""

        self.model_max_length = model_max_length
        self._token_to_id: dict[str, int] = {"[CLS]": 1, "[SEP]": 2}
        self._id_to_token: dict[int, str] = {1: "[CLS]", 2: "[SEP]"}

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
    ) -> list[int]:
        """Tokenize bằng regex và không truncate dù caller truyền cờ."""

        del truncation
        ids = [self._id_for_token(token) for token in re.findall(r"\w+|[^\w\s]", text, re.UNICODE)]
        return [1, *ids, 2] if add_special_tokens else ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """Decode gần đúng để test window splitting mà không cần download model."""

        del clean_up_tokenization_spaces
        tokens = [self._id_to_token.get(token_id, "") for token_id in token_ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in {"[CLS]", "[SEP]"}]
        text = " ".join(token for token in tokens if token)
        return re.sub(r"\s+([,.;:!?%)])", r"\1", text).strip()

    def _id_for_token(self, token: str) -> int:
        """Cấp ID ổn định trong vòng đời tokenizer để decode ngược được."""

        if token not in self._token_to_id:
            token_id = len(self._token_to_id) + 1
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return self._token_to_id[token]


def _flatten_token_ids(value: Any) -> list[int]:
    """Chuẩn hoá list/tensor/nested batch tokenizer output thành list ID phẳng."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        try:
            value = list(value)
        except TypeError as exc:
            raise TokenizerResolutionError("Tokenizer trả output input_ids không hợp lệ") from exc
    return [int(token_id) for token_id in value]

