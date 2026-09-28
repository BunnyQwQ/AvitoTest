"""Отбор кандидатов и расчёт признаков для ранкера.

Для каждого запроса скоры считаются сразу по всему корпусу (numpy / GPU), затем
берётся объединение топов нескольких источников:
  * BM25F по леммам — точные совпадения слов;
  * эмбеддинги — синонимы и смысл;
  * символьные n-граммы — опечатки и слитное написание;
  * классификатор подкатегорий — запросы без общих слов с объявлением;
  * объявления, которые уже выбирали по такому же тексту.
К текстовому скору каждого источника прибавляется одна и та же «база»:
0.5·log P(локация объявления | локация поиска) + 2·[фильтр выполнен] — так локация
и фильтр учитываются мягко, без жёсткого отсечения. Коэффициенты подобраны на валидации.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .features.filters import FilterMatcher
from .features.history import HistoryStats
from .features.locations import LocationModel
from .features.query_clf import QueryAttrClassifier
from .retrieval.bm25 import BM25F, FIELDS
from .retrieval.chargram import CharIndex

FIELD_SHORT = {"item_title_raw": "title", "item_infm_params_text": "params", "item_description_raw": "desc"}


@dataclass
class GenConfig:
    k_bm25: int = 300
    k_dense: int = 300
    k_char: int = 100
    k_mc: int = 100
    loc_w: float = 0.5          # вес log P(локация) в источниках
    flt_w: float = 2.0          # бонус за выполненный фильтр
    dense_tau: float = 20.0     # масштаб косинуса эмбеддингов (подобран для дообученной e5)
    char_scale: float = 8.0     # масштаб косинуса n-грамм
    batch: int = 256
    device: str = "cuda"        # где считать косинусы эмбеддингов ("cpu", если GPU занят)


@dataclass
class Resources:
    items: pd.DataFrame
    bm25: BM25F
    char: CharIndex
    loc: LocationModel
    fm: FilterMatcher
    clf: QueryAttrClassifier
    hist: HistoryStats
    item_emb: np.ndarray                      # (N × d) float16, L2-нормированные
    static: dict = field(default_factory=dict)

    def __post_init__(self):
        it = self.items
        self.static = {
            "log_price": np.log1p(it["item_price"].fillna(-1).clip(lower=0).to_numpy(np.float32)),
            "price_missing": it["item_price"].isna().to_numpy(np.float32),
            "rating": it["item_rating"].fillna(-1).to_numpy(np.float32),
            "log_reviews": np.log1p(it["item_rating_reviews_count"].fillna(0).to_numpy(np.float32)),
            "reviews_missing": it["item_rating_reviews_count"].isna().to_numpy(np.float32),
            "phone_hidden": it["item_is_phone_hidden"].astype(np.float32).to_numpy(),
            "msg_forbidden": it["item_is_message_forbidden"].astype(np.float32).to_numpy(),
            "title_len": it["item_title_raw"].str.len().to_numpy(np.float32),
            "desc_len": np.log1p(it["item_description_raw"].str.len().to_numpy(np.float32)),
            "params_len": np.log1p(it["item_infm_params_text"].str.len().to_numpy(np.float32)),
        }
        # нормализованные заголовки для признака «запрос целиком входит в заголовок»
        self.title_norm = np.array([" ".join(str(s).lower().replace("ё", "е").split()) for s in it["item_title_raw"]],
                                   dtype=object)
        self.mc_items = it["item_microcat_id"].to_numpy()
        self.cat_items = it["item_category_id"].to_numpy()
        self.item_ids = it["item_id"].to_numpy()
        # сколько объявлений корпуса в каждой локации (размер «рынка»)
        self.loc_size = it.groupby("item_location_id")["item_id"].transform("size").to_numpy(np.float32)


def _topk(v: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(v) - 1)
    idx = np.argpartition(-v, k)[:k]
    return idx[np.argsort(-v[idx], kind="stable")]


def _row_dense(M: sp.csr_matrix, r: int, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    s, e = M.indptr[r], M.indptr[r + 1]
    out[M.indices[s:e]] = M.data[s:e]
    return out


def generate(queries: pd.DataFrame, allowed: dict, R: Resources, q_emb: np.ndarray, P_mc: np.ndarray,
             gold: dict | None = None, cfg: GenConfig = GenConfig()) -> pd.DataFrame:
    """Кандидаты + признаки для набора запросов.

    queries — query_id, qn, flt, search_location_id, search_category, chunk;
    allowed — chunk -> булев вектор «объявление входит в корпус этой порции»;
    q_emb   — эмбеддинги запросов (в порядке queries); P_mc — вероятности подкатегорий.
    """
    import torch

    N = len(R.items)
    if cfg.device == "cuda":
        item_emb_t = torch.from_numpy(R.item_emb).cuda()
    else:
        item_emb_t = torch.from_numpy(R.item_emb.astype(np.float32))
        q_emb = q_emb.astype(np.float32)
    logrev = R.static["log_reviews"]
    mc_cls = R.clf.class_index(R.mc_items)          # один раз на корпус
    q = queries.reset_index(drop=True)
    Qmat = R.bm25.query_matrix(q["qn"].tolist())
    # Идём в порядке (порция, локация поиска): векторы локации переиспользуются
    order = np.lexsort((q["search_location_id"].to_numpy(), q["chunk"].to_numpy()))
    out_frames = []
    cur_loc, lf, llp = None, None, None
    for b0 in range(0, len(q), cfg.batch):
        bidx = order[b0:b0 + cfg.batch]
        S_bm = R.bm25.scores(Qmat[bidx])
        S_f = {f: R.bm25.scores(Qmat[bidx], field=f) for f in FIELDS}
        S_ch = R.char.scores(q["qn"].iloc[bidx].tolist())
        with torch.no_grad():
            D = (torch.from_numpy(q_emb[bidx]).to(item_emb_t.device) @ item_emb_t.T).float().cpu().numpy()
        for r, qi in enumerate(bidx):
            row = q.iloc[qi]
            alw = allowed[row["chunk"]]
            sl = int(row["search_location_id"])
            if sl != cur_loc:
                cur_loc, lf = sl, R.loc.features(sl)
                llp = np.log(lf["loc_p"] + 1e-4).astype(np.float32)
            ff = R.fm.features(row["flt"])
            base = cfg.loc_w * llp
            if ff is not None:
                base = base + cfg.flt_w * ff["flt_all"]
            base = np.where(alw, base, -1e9).astype(np.float32)

            bm = _row_dense(S_bm, r, N)
            ch = _row_dense(S_ch, r, N)
            de = D[r]
            pmc = R.clf.proba_for(P_mc, qi, mc_cls)

            srcs = {
                "bm25": _topk(bm + base, cfg.k_bm25),
                "dense": _topk(cfg.dense_tau * de + base, cfg.k_dense),
                "char": _topk(cfg.char_scale * ch + base, cfg.k_char),
                "mc": _topk(np.log(pmc + 1e-6) + base + 0.1 * logrev, cfg.k_mc),
            }
            ti = R.hist.text_items.get(row["qn"])
            hist_idx = ti[0][alw[ti[0]]] if ti is not None else np.zeros(0, dtype=np.int64)
            cand = np.unique(np.concatenate([*srcs.values(), hist_idx]))
            cand = cand[alw[cand]]
            if len(cand) == 0:
                continue

            feats = {"query_id": np.full(len(cand), row["query_id"]), "item_idx": cand}
            # ранг кандидата в каждом источнике (len(top), если источник его не выбрал)
            for name, top in srcs.items():
                pos = {i: j for j, i in enumerate(top)}
                feats[f"rank_{name}"] = np.array([pos.get(i, len(top)) for i in cand], dtype=np.float32)
            feats["bm25"] = bm[cand]
            for f in FIELDS:
                feats["bm25_" + FIELD_SHORT[f]] = _row_dense(S_f[f], r, N)[cand]
            qn = row["qn"]
            feats["q_in_title"] = np.array([qn in t for t in R.title_norm[cand]], dtype=np.float32)
            feats["char_sim"] = ch[cand]
            feats["dense_sim"] = de[cand]
            feats["p_mc"] = pmc[cand]
            feats["mc_logp"] = R.hist.mc_logp[cand]
            feats["mc_sameloc"] = R.hist.mc_sameloc[cand]
            lmc = R.hist.loc_mc_logp.get(sl)
            if lmc is not None:
                feats["loc_mc_logp"] = (pd.Series(R.mc_items[cand]).map(lmc)
                                        .fillna(R.hist.loc_mc_default[sl]).to_numpy(np.float32))
            else:
                feats["loc_mc_logp"] = np.full(len(cand), np.nan, np.float32)
            feats["loc_p"] = lf["loc_p"][cand].astype(np.float32)
            feats["same_loc"] = lf["same_loc"][cand].astype(np.float32)
            feats["dist_km"] = np.log1p(np.nan_to_num(lf["dist_km"][cand], nan=-1).clip(min=0)).astype(np.float32)
            feats["dist_missing"] = np.isnan(lf["dist_km"][cand]).astype(np.float32)
            feats["loc_size"] = np.log1p(R.loc_size[cand])
            if ff is not None:
                for k in ("flt_n_keys", "flt_matched", "flt_all", "flt_vid", "flt_tip"):
                    feats[k] = ff[k][cand].astype(np.float32)
            else:
                for k in ("flt_n_keys", "flt_matched"):
                    feats[k] = np.zeros(len(cand), np.float32)
                for k in ("flt_all", "flt_vid", "flt_tip"):
                    feats[k] = np.full(len(cand), -1, np.float32)     # «фильтра нет» ≠ «фильтр не выполнен»
            cat = int(row["search_category"])
            feats["cat_ok"] = ((cat == 0) | (R.cat_items[cand] == cat)).astype(np.float32)
            feats["query_cat0"] = np.full(len(cand), float(cat == 0), np.float32)
            # покрытие слов запроса полями объявления (доля и доля с весом IDF)
            qterms = Qmat[qi].indices
            nt = len(qterms)
            if nt:
                idf = R.bm25.idf[qterms]
                any_cov = np.zeros((len(cand), nt), dtype=bool)
                for f in FIELDS:
                    pres = (R.bm25.tf[f][cand][:, qterms].toarray() > 0)
                    any_cov |= pres
                    nm = FIELD_SHORT[f]
                    feats["cov_" + nm] = pres.mean(1).astype(np.float32)
                    feats["idfcov_" + nm] = (pres @ idf / idf.sum()).astype(np.float32)
                feats["cov_any"] = any_cov.mean(1).astype(np.float32)
                feats["idfcov_any"] = (any_cov @ idf / idf.sum()).astype(np.float32)
            else:
                for nm in ("title", "params", "desc", "any"):
                    feats["cov_" + nm] = np.zeros(len(cand), np.float32)
                    feats["idfcov_" + nm] = np.zeros(len(cand), np.float32)
            # история выборов
            feats["hist_item_cnt"] = np.log1p(R.hist.item_cnt[cand])
            feats["hist_item_nq"] = np.log1p(R.hist.item_nq[cand])
            ht = np.zeros(len(cand), np.float32)
            if ti is not None:
                m = dict(zip(ti[0], ti[1]))
                ht = np.array([m.get(i, 0.0) for i in cand], dtype=np.float32)
            feats["hist_same_text"] = ht
            for k, v in R.static.items():
                feats[k] = v[cand]
            # признаки запроса (одинаковые у всех кандидатов) и относительные
            feats["q_nterms"] = np.full(len(cand), nt, np.float32)
            feats["q_len"] = np.full(len(cand), len(row["qn"]), np.float32)
            feats["q_seen"] = np.full(len(cand), float(row["qn"] in R.hist.texts), np.float32)
            feats["q_has_filter"] = np.full(len(cand), float(ff is not None), np.float32)
            n_same = int((lf["same_loc"] & alw).sum())       # объявлений корпуса в самой локации поиска
            feats["q_region"] = np.full(len(cand), float(n_same == 0), np.float32)
            feats["q_n_sameloc"] = np.full(len(cand), np.log1p(n_same), np.float32)
            feats["bm25_rel"] = feats["bm25"] / (feats["bm25"].max() + 1e-6)
            feats["dense_rel"] = feats["dense_sim"] - feats["dense_sim"].max()
            feats["char_rel"] = feats["char_sim"] / (feats["char_sim"].max() + 1e-6)
            feats["n_cand"] = np.full(len(cand), len(cand), np.float32)
            if gold is not None:
                g = gold.get(row["query_id"], set())
                feats["label"] = np.array([R.item_ids[i] in g for i in cand], dtype=np.int8)
            out_frames.append(pd.DataFrame(feats))
    return pd.concat(out_frames, ignore_index=True)
