"""Поиск по символьным n-граммам заголовка.

Ловит то, что не ловят леммы: опечатки («масаж», «кортон»), слитное написание
(«грузтакси»), однокоренные слова, которые pymorphy не сводит («токарь» / «токарные»).
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _prep(s: str) -> str:
    return " ".join(str(s).lower().replace("ё", "е").split())


class CharIndex:
    def __init__(self, items, ngram_range=(3, 5), field: str = "item_title_raw"):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, min_df=2,
                                   sublinear_tf=True, dtype=np.float32)
        X = self.vec.fit_transform([_prep(s) for s in items[field].values])
        self.XT = X.T.tocsr()              # n-граммы × объявления

    def scores(self, queries):
        """Косинус «запрос — заголовок» для пачки запросов (разреженная матрица)."""
        Q = self.vec.transform([_prep(q) for q in queries])
        return (Q @ self.XT).tocsr()
