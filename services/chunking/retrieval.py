"""Child retrieval, per-parent deduplication và adaptive parent/sibling expansion."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Any

from .models import Chunk, RetrievalResult


def deduplicate_children(
    retrieved_children: Iterable[Any],
    max_children_per_parent: int = 3,
) -> list[Any]:
    """Giữ ranking gốc nhưng giới hạn số hit cùng parent để tăng document diversity."""

    if max_children_per_parent <= 0:
        return list(retrieved_children)
    counts: Counter[str] = Counter()
    output: list[Any] = []
    for result in retrieved_children:
        payload = _payload(result)
        parent_id = str(payload.get("parent_id") or payload.get("chunk_id") or id(result))
        if counts[parent_id] >= max_children_per_parent:
            continue
        counts[parent_id] += 1
        output.append(result)
    return output


def expand_context(
    retrieved_children: Iterable[Any],
    parent_store: Mapping[str, Any],
    *,
    sibling_store: Mapping[str, Iterable[Any]] | None = None,
    strategy: str = "adaptive",
) -> list[RetrievalResult]:
    """Mở rộng context theo ``none|parent|siblings|adaptive``.

    Adaptive giữ table row chính xác và ưu tiên table schema parent; prose hoặc
    nhiều hit cùng article sẽ thêm full structural parent.
    """

    if strategy not in {"none", "parent", "siblings", "adaptive"}:
        raise ValueError(f"Expansion strategy không hợp lệ: {strategy}")
    children = list(retrieved_children)
    hit_counts = Counter(str(_payload(item).get("parent_id") or "") for item in children)
    output: list[RetrievalResult] = []
    for item in children:
        payload = _payload(item)
        parent_id = str(payload.get("parent_id") or "")
        contexts: list[dict[str, Any]] = []

        if strategy == "parent":
            _append_store_record(contexts, parent_store.get(parent_id))
        elif strategy == "siblings" and sibling_store:
            for sibling in sibling_store.get(parent_id, []):
                sibling_payload = _payload(sibling)
                if sibling_payload.get("chunk_id") != payload.get("chunk_id"):
                    contexts.append(sibling_payload)
        elif strategy == "adaptive":
            if payload.get("chunk_type") == "table_row":
                table_parent_id = str(payload.get("table_parent_id") or "")
                _append_store_record(contexts, parent_store.get(table_parent_id))
                if hit_counts[parent_id] > 1 and not contexts:
                    _append_store_record(contexts, parent_store.get(parent_id))
            else:
                _append_store_record(contexts, parent_store.get(parent_id))

        output.append(
            RetrievalResult(
                score=_score(item),
                child=payload,
                context=contexts,
            )
        )
    return output


def embedding_search_v2(
    client: Any,
    embedding_model: Any,
    collection_name: str,
    query: str,
    *,
    top_k: int = 10,
    expand_context_enabled: bool = True,
    parent_store: Mapping[str, Any] | None = None,
    max_children_per_parent: int = 3,
) -> list[RetrievalResult]:
    """Search chỉ child records, deduplicate và tùy chọn adaptive context expansion."""

    from qdrant_client.models import FieldCondition, Filter, MatchValue

    query_vector = embedding_model.encode(query, normalize_embeddings=True)
    vector = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
    child_filter = Filter(
        must=[FieldCondition(key="record_type", match=MatchValue(value="child"))]
    )
    response = client.query_points(
        collection_name=collection_name,
        query=vector,
        query_filter=child_filter,
        limit=max(top_k, top_k * max(1, max_children_per_parent)),
    )
    points = getattr(response, "points", response)
    deduplicated = deduplicate_children(points, max_children_per_parent)[:top_k]
    strategy = "adaptive" if expand_context_enabled and parent_store else "none"
    return expand_context(deduplicated, parent_store or {}, strategy=strategy)


def build_sibling_store(chunks: Iterable[Chunk]) -> dict[str, list[dict[str, Any]]]:
    """Nhóm child chunks theo parent để strategy ``siblings`` dùng offline."""

    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        if chunk.metadata.get("record_type") == "child" and chunk.parent_id:
            result[chunk.parent_id].append(_payload(chunk))
    return dict(result)


def _payload(item: Any) -> dict[str, Any]:
    """Chuẩn hoá Qdrant result/dict/Chunk thành payload dictionary."""

    if isinstance(item, Chunk):
        value = asdict(item)
        value.update(value.pop("metadata", {}))
        return value
    if isinstance(item, dict):
        if isinstance(item.get("payload"), dict):
            return dict(item["payload"])
        return dict(item)
    payload = getattr(item, "payload", None)
    return dict(payload or {})


def _score(item: Any) -> float | None:
    """Lấy score từ Qdrant object/dict nếu có."""

    if isinstance(item, dict):
        value = item.get("score")
    else:
        value = getattr(item, "score", None)
    return float(value) if value is not None else None


def _append_store_record(output: list[dict[str, Any]], record: Any) -> None:
    """Append parent store record nếu tồn tại và không trùng context."""

    if record is None:
        return
    payload = _payload(record)
    if payload and payload not in output:
        output.append(payload)
