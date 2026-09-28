"""Ранжирующая модель второго уровня (LightGBM LambdaRank).

Кандидатов ~500 на запрос; модель должна поднять правильные объявления в
топ-50. LambdaRank оптимизирует порядок внутри запроса (группы), что ближе
к целевой метрике Recall@50, чем независимая бинарная классификация.
Запросы, у которых среди кандидатов нет ни одного правильного ответа, в
обучении бесполезны (нечего поднимать) — их отбрасываем.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C

NON_FEATURES = {"query_id", "item_idx", "label", "chunk"}

# Признаки, построенные на рубрикаторе Услуг (подкатегория и её статистики, длина
# параметров услуги). В train правильные ответы почти всегда из категории 114,
# поэтому через эти признаки модель выучивает «не-услуга -> плохо». Для поиска по
# всем категориям (search_category = 0, 9% бенчмарка) это системная ошибка, поэтому
# для таких запросов используется модель без них. cat_ok и фильтры оставляем:
# у запросов категории 0 cat_ok = 1 для всех объявлений, фильтров у них нет.
CATEGORY_SPECIFIC = ["p_mc", "mc_logp", "rank_mc", "mc_sameloc", "loc_mc_logp", "params_len"]

DEFAULT_PARAMS = dict(
    objective="lambdarank", metric="ndcg", eval_at=[50], lambdarank_truncation_level=100,
    learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=C.SEED,
    deterministic=True, force_row_wise=True, num_threads=16,
)


def feature_cols(df: pd.DataFrame, exclude=()) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES and c not in set(exclude)]


def _prep(df: pd.DataFrame, drop_empty: bool):
    if drop_empty:
        has_pos = df.groupby("query_id")["label"].transform("max") > 0
        df = df[has_pos]
    df = df.sort_values(["query_id"], kind="stable")
    groups = df.groupby("query_id", sort=False).size().to_numpy()
    return df, groups


def train(train_df: pd.DataFrame, valid_df: pd.DataFrame | None = None, params: dict | None = None,
          num_boost_round: int = 3000, early_stopping: int = 100, exclude=()):
    params = {**DEFAULT_PARAMS, **(params or {})}
    feats = feature_cols(train_df, exclude)
    tr, g_tr = _prep(train_df, drop_empty=True)
    dtr = lgb.Dataset(tr[feats], tr["label"], group=g_tr, free_raw_data=True)
    valid_sets, callbacks = [], [lgb.log_evaluation(100)]
    if valid_df is not None:
        va, g_va = _prep(valid_df, drop_empty=True)
        valid_sets = [lgb.Dataset(va[feats], va["label"], group=g_va, reference=dtr)]
        callbacks.append(lgb.early_stopping(early_stopping, verbose=True))
    model = lgb.train(params, dtr, num_boost_round=num_boost_round, valid_sets=valid_sets, callbacks=callbacks)
    return model, feats


def predict_topk(model, feats, df: pd.DataFrame, item_ids: np.ndarray, k: int = C.TOP_K) -> dict:
    """query_id -> список item_id топ-k по скору модели."""
    s = model.predict(df[feats], num_threads=16)
    tmp = pd.DataFrame({"query_id": df["query_id"].to_numpy(), "item_idx": df["item_idx"].to_numpy(), "s": s})
    tmp = tmp.sort_values(["query_id", "s"], ascending=[True, False], kind="stable")
    tmp = tmp.groupby("query_id", sort=False).head(k)
    return {q: item_ids[g.to_numpy()].tolist() for q, g in tmp.groupby("query_id", sort=False)["item_idx"]}
