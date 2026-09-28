"""Метрика соревнования: Recall@K, усреднённый по запросам."""
from __future__ import annotations

import numpy as np


def recall_at_k(pred: dict, gold: dict, k: int = 50) -> float:
    """pred: query_id -> упорядоченный список item_id; gold: query_id -> множество item_id."""
    vals = []
    for qid, rel in gold.items():
        top = set(pred.get(qid, [])[:k])
        vals.append(len(top & rel) / len(rel))
    return float(np.mean(vals))


def recall_curve(pred: dict, gold: dict, ks=(50, 100, 200, 300, 500, 1000)) -> dict:
    return {k: round(recall_at_k(pred, gold, k), 4) for k in ks}
