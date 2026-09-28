"""Сопоставление фильтров поиска с параметрами объявления.

Фильтр запроса — склейка пар «ключ значение», например
    «Тип услуги Маникюр, педикюр Вид услуги Красота, здоровье».
В параметрах объявления те же пары записаны в том же формате, поэтому проверка
сводится к поиску подстроки «ключ значение».

EDA: для «Вид услуги» значение фильтра есть у 98.4% выбранных объявлений,
для «Тип услуги» — у 97.8%, т.е. это почти жёсткие ограничения. Ключи вроде
«Рейтинг пользователя» или «Поиск по слотам» в параметрах объявления не
отражены — их не проверяем.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

# Все известные ключи фильтров (длинные раньше коротких, чтобы
# «Тип услуги автосервиса» не разбирался как «Тип услуги»).
KEYS = ["Тип услуги автосервиса", "Вид услуги", "Тип услуги", "Онлайн-запись", "Кто оказывает услуги",
        "Аренда авто", "Предмет или специальность", "Поиск по слотам", "Срочная услуга (мультистатус)",
        "Участие в пилоте CPL", "Рейтинг пользователя", "Ваши клиенты", "Сортировка для URL",
        "Чем вы занимаетесь", "Где вы оказываете услуги", "Специальность или сфера", "Опыт работы"]
# Ключи, значения которых можно проверить по параметрам объявления
CHECKABLE = {"Тип услуги автосервиса", "Вид услуги", "Тип услуги", "Аренда авто", "Предмет или специальность",
             "Чем вы занимаетесь", "Ваши клиенты", "Где вы оказываете услуги", "Специальность или сфера"}

_KEY_RE = re.compile("(" + "|".join(re.escape(k) for k in sorted(KEYS, key=len, reverse=True)) + ")")


def parse_filter(s: str) -> dict[str, list[str]]:
    """'Тип услуги A Вид услуги B' -> {'Тип услуги': ['A'], 'Вид услуги': ['B']} (только проверяемые ключи)."""
    parts = _KEY_RE.split(s or "")
    out: dict[str, list[str]] = defaultdict(list)
    for i in range(1, len(parts), 2):
        key, val = parts[i], parts[i + 1].strip()
        if key in CHECKABLE and val:
            out[key].append(val)
    return dict(out)


class FilterMatcher:
    """Кэширует булевы векторы «объявление содержит пару ключ-значение»."""

    WINDOW = 80   # сколько символов после ключа хватает для любого значения фильтра

    def __init__(self, items: pd.DataFrame):
        params = items["item_infm_params_text"].fillna("").tolist()
        # Для ускорения ищем подстроку не во всём тексте параметров (~1000 символов),
        # а в коротких «окнах» после каждого вхождения ключа — в ~10 раз быстрее.
        self.windows: dict[str, pd.Series] = {}
        for key in CHECKABLE:
            pat = key + " "
            wins = []
            for p in params:
                parts, start = [], p.find(pat)
                while start >= 0:
                    parts.append(p[start:start + len(pat) + self.WINDOW])
                    start = p.find(pat, start + 1)
                wins.append(" || ".join(parts))
            self.windows[key] = pd.Series(wins)
        self.params = pd.Series(params)
        self._pair_cache: dict[str, np.ndarray] = {}
        self._filter_cache: dict[str, dict[str, np.ndarray]] = {}

    def _pair_vec(self, key: str, val: str) -> np.ndarray:
        s = f"{key} {val}"
        if s not in self._pair_cache:
            # длинные значения (редкие «склеенные» фильтры автосервиса) в окно не влезают —
            # для них честный поиск по всему тексту параметров
            src = self.windows[key] if len(val) <= self.WINDOW else self.params
            self._pair_cache[s] = src.str.contains(s, regex=False).to_numpy()
        return self._pair_cache[s]

    def features(self, flt: str) -> dict[str, np.ndarray] | None:
        """Признаки по всем объявлениям для одного фильтра (None, если проверять нечего).

        flt_n_keys   — сколько проверяемых ключей в фильтре;
        flt_matched  — сколько из них выполнено (значения одного ключа — через «ИЛИ»);
        flt_all      — выполнены все проверяемые ключи;
        flt_vid/flt_tip — отдельно «Вид услуги» и «Тип услуги» (самые частые).
        """
        if flt in self._filter_cache:
            return self._filter_cache[flt]
        parsed = parse_filter(flt)
        if not parsed:
            self._filter_cache[flt] = None
            return None
        n = len(self.params)
        matched = np.zeros(n, dtype=np.int8)
        per_key = {}
        for key, vals in parsed.items():
            v = np.zeros(n, dtype=bool)
            for val in vals:
                v |= self._pair_vec(key, val)
            per_key[key] = v
            matched += v
        res = {"flt_n_keys": np.full(n, len(parsed), dtype=np.int8), "flt_matched": matched,
               "flt_all": matched == len(parsed),
               "flt_vid": per_key.get("Вид услуги", np.ones(n, dtype=bool)),
               "flt_tip": per_key.get("Тип услуги", np.ones(n, dtype=bool))}
        self._filter_cache[flt] = res
        return res
