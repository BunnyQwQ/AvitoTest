"""Классификатор текст запроса -> подкатегория объявления (item_microcat_id).

Зачем: даже когда у запроса и объявления нет общих слов ("бровки" / "Мастер
бровист"), история выборов подсказывает, в какую подкатегорию обычно ведёт
такой запрос. По EDA у одного текста запроса в среднем 82% выборов приходится
на одну подкатегорию. Для кандидата отсюда получается признак P(подкатегория объявления | запрос).

Модель - логистическая регрессия на TF-IDF:
  * леммы (униграммы + биграммы), чтобы не зависеть от словоформы;
  * символьные n-граммы 2-5 для устойчивости к опечаткам ("масаж", "оьучение").
Учу на уникальных парах (текст, подкатегория) из истории с весом = число выборов,
так в разы быстрее, чем на сырых строках.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .. import text as T


def _lemma_str(q: str) -> str:
    return " ".join(T.lemmas(q, drop_stop=True))


class QueryAttrClassifier:
    def __init__(self, C: float = 10.0, max_iter: int = 200):
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"\S+")
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300_000)
        self.model = LogisticRegression(C=C, max_iter=max_iter)

    def _X(self, texts, fit=False):
        lem = [_lemma_str(t) for t in texts]
        if fit:
            return sp.hstack([self.word_vec.fit_transform(lem), self.char_vec.fit_transform(texts)]).tocsr()
        return sp.hstack([self.word_vec.transform(lem), self.char_vec.transform(texts)]).tocsr()

    def fit(self, history: pd.DataFrame, target: str = "item_microcat_id"):
        agg = history.groupby(["qn", target]).size().rename("w").reset_index()
        X = self._X(agg["qn"].tolist(), fit=True)
        self.model.fit(X, agg[target].to_numpy(), sample_weight=agg["w"].to_numpy())
        self.classes_ = self.model.classes_
        self.class_pos = {c: i for i, c in enumerate(self.classes_)}
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return self.model.predict_proba(self._X(list(texts)))

    def class_index(self, attr_values: np.ndarray) -> np.ndarray:
        """Номер класса для каждого значения атрибута (-1, если такого класса модель не видела).
        Считается один раз на корпус, дальше P(attr | запрос) берётся простым индексированием."""
        return np.array([self.class_pos.get(v, -1) for v in attr_values], dtype=np.int64)

    @staticmethod
    def proba_for(P: np.ndarray, row: int, cls_idx: np.ndarray) -> np.ndarray:
        """P(attr объявления | запрос row) по всем объявлениям (0 для неизвестных классов)."""
        return np.where(cls_idx >= 0, P[row, np.maximum(cls_idx, 0)], 0.0).astype(np.float32)
