"""BM25F по лемматизированному тексту объявления (заголовок, параметры, описание).

Поля складываются с весами до насыщения (BM25F):
    tf~(d,t) = Σ_f w_f · tf_f(d,t) / (1 - b_f + b_f · len_f(d) / avglen_f)
    score(q,d) = Σ_{t∈q} idf(t) · tf~ / (k1 + tf~)
Всё хранится в разреженных матрицах, так что скоры запроса по всему корпусу —
одно матричное умножение.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .. import text as T

FIELDS = ("item_title_raw", "item_infm_params_text", "item_description_raw")


class Vocab:
    def __init__(self):
        self.idx: dict[str, int] = {}

    def ids(self, toks, grow: bool):
        out = []
        for t in toks:
            i = self.idx.get(t)
            if i is None and grow:
                i = self.idx[t] = len(self.idx)
            if i is not None:
                out.append(i)
        return out


def _tf_matrix(docs_ids, n_vocab: int) -> sp.csr_matrix:
    indptr = np.zeros(len(docs_ids) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(d) for d in docs_ids])
    indices = np.fromiter((i for d in docs_ids for i in d), dtype=np.int32, count=indptr[-1])
    m = sp.csr_matrix((np.ones(len(indices), dtype=np.float32), indices, indptr), shape=(len(docs_ids), n_vocab))
    m.sum_duplicates()          # повторы слова в документе складываются в частоту
    return m


class BM25F:
    def __init__(self, items, weights=(3.0, 1.0, 1.0), bs=(0.3, 0.75, 0.75), k1=1.2):
        self.vocab = Vocab()
        docs = {f: [self.vocab.ids(T.lemmas(s), grow=True) for s in items[f].values] for f in FIELDS}
        V = len(self.vocab.idx)
        self.tf = {f: _tf_matrix(docs[f], V) for f in FIELDS}
        self.lens = {f: np.asarray(self.tf[f].sum(axis=1)).ravel() for f in FIELDS}
        n = len(items)
        # document frequency считаю по «любому полю»
        any_field = sum((self.tf[f] > 0).astype(np.float32) for f in FIELDS)
        df = np.asarray((any_field > 0).sum(axis=0)).ravel()
        self.idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.n_items = n
        self.set_params(weights, bs, k1)

    def _field_norm(self, f, b):
        L = self.lens[f]
        return 1.0 / (1.0 - b + b * L / max(L.mean(), 1e-9))

    def set_params(self, weights, bs, k1):
        tfc = None
        for f, w, b in zip(FIELDS, weights, bs):
            part = sp.diags(self._field_norm(f, b).astype(np.float32)) @ self.tf[f] * w
            tfc = part if tfc is None else tfc + part
        tfc = tfc.tocsr()
        tfc.data = tfc.data / (k1 + tfc.data)
        W = tfc @ sp.diags(self.idf)
        self.WT = W.T.tocsr().astype(np.float32)        # слова × объявления
        # BM25 по каждому полю отдельно — пойдут признаками в ранкер
        self.field_WT = {}
        for f, b in zip(FIELDS, bs):
            m = (sp.diags(self._field_norm(f, b).astype(np.float32)) @ self.tf[f]).tocsr()
            m.data = m.data / (k1 + m.data)
            self.field_WT[f] = (m @ sp.diags(self.idf)).T.tocsr().astype(np.float32)

    def query_matrix(self, queries) -> sp.csr_matrix:
        """Запросы × слова, каждое слово запроса учитывается один раз."""
        rows = [sorted(set(self.vocab.ids(T.lemmas(q, drop_stop=True), grow=False))) for q in queries]
        indptr = np.zeros(len(rows) + 1, dtype=np.int64)
        indptr[1:] = np.cumsum([len(r) for r in rows])
        indices = np.fromiter((i for r in rows for i in r), dtype=np.int32, count=indptr[-1])
        return sp.csr_matrix((np.ones(len(indices), dtype=np.float32), indices, indptr),
                             shape=(len(rows), len(self.vocab.idx)))

    def scores(self, Q: sp.csr_matrix, field: str | None = None) -> sp.csr_matrix:
        WT = self.WT if field is None else self.field_WT[field]
        return (Q @ WT).tocsr()
