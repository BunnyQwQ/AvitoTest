"""Модель локаций: насколько объявление из локации L подходит поиску из локации S.

Факты из EDA:
  * 83% выборов — объявление из той же локации, что и поиск;
  * при несовпадении это пригород/соседний город (медиана ~16 км от центра локации
    поиска, 2/3 — ближе 30 км) или поиск по «региону»
    (например, локация 107620 — видимо, «Москва и область»: своих объявлений
    у неё нет, 61% выборов уходит в Москву).

Поэтому вместо жёсткого фильтра «та же локация» используем:
  * log P(локация объявления | локация поиска) по истории выборов со сглаживанием;
  * расстояние от «центра» локации поиска до объявления. Центр локации поиска —
    медиана координат выбранных из неё объявлений (работает и для регионов).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class LocationModel:
    """Считает признаки «локация поиска × объявление» для всех объявлений сразу."""

    def __init__(self, history: pd.DataFrame, items: pd.DataFrame, alpha: float = 1.0):
        # Центроиды локаций объявлений (по всем известным объявлениям)
        all_it = pd.concat([items[["item_id", "item_location_id", "item_latitude", "item_longitude"]],
                            history[["item_id", "item_location_id", "item_latitude", "item_longitude"]]]
                           ).drop_duplicates("item_id")
        self.item_loc_centroid = all_it.groupby("item_location_id")[["item_latitude", "item_longitude"]].median()
        self.item_loc_size = all_it.groupby("item_location_id").size()

        # Центр локации поиска = медиана координат выбранных объявлений
        sc = history.groupby("search_location_id")[["item_latitude", "item_longitude"]].median()
        # запасной вариант — центроид одноимённой локации объявлений
        fb = self.item_loc_centroid.loc[~self.item_loc_centroid.index.isin(sc.index)]
        self.search_centroid = pd.concat([sc, fb])

        # Переходы «локация поиска → локация объявления»
        tr = history.groupby(["search_location_id", "item_location_id"]).size().rename("n").reset_index()
        tot = tr.groupby("search_location_id")["n"].sum()
        tr["p"] = tr["n"] / tr["search_location_id"].map(tot)
        self.trans = tr
        self.search_total = tot
        self.alpha = alpha

        # Индексы локаций объявлений в корпусе для быстрых векторных операций
        self.items_loc = items["item_location_id"].to_numpy()
        self.items_lat = items["item_latitude"].to_numpy()
        self.items_lon = items["item_longitude"].to_numpy()
        self.loc_codes, self.items_loc_idx = np.unique(self.items_loc, return_inverse=True)
        self._p_cache: dict[int, np.ndarray] = {}

    def trans_prob_vec(self, search_loc: int) -> np.ndarray:
        """P(локация объявления | локация поиска) для каждой УНИКАЛЬНОЙ локации корпуса.

        Сглаживание: к счётчикам истории добавляется «псевдо-выбор» своей же локации,
        чтобы у редких локаций поиска без истории своя локация всё равно была вероятной.
        """
        if search_loc in self._p_cache:
            return self._p_cache[search_loc]
        p = np.zeros(len(self.loc_codes), dtype=np.float64)
        sub = self.trans[self.trans["search_location_id"] == search_loc]
        n_tot = float(self.search_total.get(search_loc, 0.0))
        if len(sub):
            pos = np.searchsorted(self.loc_codes, sub["item_location_id"].to_numpy())
            ok = (pos < len(self.loc_codes)) & (self.loc_codes[np.minimum(pos, len(self.loc_codes) - 1)] == sub["item_location_id"].to_numpy())
            np.add.at(p, pos[ok], sub["n"].to_numpy()[ok])
        own = np.searchsorted(self.loc_codes, search_loc)
        if own < len(self.loc_codes) and self.loc_codes[own] == search_loc:
            p[own] += self.alpha
            n_tot += self.alpha
        p = p / n_tot if n_tot > 0 else p
        self._p_cache[search_loc] = p
        return p

    def features(self, search_loc: int) -> dict[str, np.ndarray]:
        """Векторы признаков по всем объявлениям корпуса для одной локации поиска."""
        p = self.trans_prob_vec(search_loc)[self.items_loc_idx]
        if search_loc in self.search_centroid.index:
            clat, clon = self.search_centroid.loc[search_loc]
            dist = haversine_km(clat, clon, self.items_lat, self.items_lon)
        else:
            dist = np.full(len(self.items_loc), np.nan)
        return {"loc_p": p, "same_loc": (self.items_loc == search_loc), "dist_km": dist}
