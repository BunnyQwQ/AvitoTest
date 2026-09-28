"""Статистики истории выборов: популярность объявлений и подкатегорий.

На валидации история = train без валидационных запросов, на бенчмарке весь train.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class HistoryStats:
    def __init__(self, history: pd.DataFrame, items: pd.DataFrame):
        id2idx = pd.Series(np.arange(len(items)), index=items["item_id"].to_numpy())
        cnt = history["item_id"].value_counts()
        nq = history.groupby("item_id")["qn"].nunique()
        self.item_cnt = items["item_id"].map(cnt).fillna(0).to_numpy(np.float32)      # сколько раз выбирали
        self.item_nq = items["item_id"].map(nq).fillna(0).to_numpy(np.float32)        # по скольким текстам
        # популярность подкатегории среди выборов
        mc = history["item_microcat_id"].value_counts(normalize=True)
        self.mc_logp = np.log(items["item_microcat_id"].map(mc).fillna(0).to_numpy(np.float32) + 1e-6)
        # доля выборов в своей же локации по подкатегориям: у онлайн-услуг она низкая,
        # и дальняя локация для них не так страшна (сглаживаю к общему среднему)
        same = (history["search_location_id"] == history["item_location_id"]).astype(np.float32)
        g = same.groupby(history["item_microcat_id"]).agg(["sum", "count"])
        prior = float(same.mean())
        rate = (g["sum"] + 20 * prior) / (g["count"] + 20)
        self.mc_sameloc = items["item_microcat_id"].map(rate).fillna(prior).to_numpy(np.float32)
        # локальный спрос: log P(подкатегория | локация поиска)
        lm = history.groupby(["search_location_id", "item_microcat_id"]).size()
        tot = history.groupby("search_location_id").size()
        self.loc_mc_logp = {sl: np.log((grp.droplevel(0) + 1.0) / (tot[sl] + 200.0))
                            for sl, grp in lm.groupby(level=0)}
        self.loc_mc_default = {sl: np.log(1.0 / (tot[sl] + 200.0)) for sl in tot.index}
        # что уже выбирали по такому же тексту запроса: qn -> (индексы объявлений, счётчики)
        h = history[history["item_id"].isin(id2idx.index)]
        g = h.groupby(["qn", "item_id"]).size().rename("n").reset_index()
        g["idx"] = g["item_id"].map(id2idx).to_numpy()
        self.text_items = {q: (d["idx"].to_numpy(), d["n"].to_numpy(np.float32)) for q, d in g.groupby("qn")}
        self.texts = set(history["qn"].unique())
