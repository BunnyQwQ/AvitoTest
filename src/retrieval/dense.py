"""Семантический поиск на эмбеддингах (multilingual-e5-base, дообученный на train).

Нужен для синонимов и смысловых связей, где слова не совпадают совсем:
"разгрузка вагонов" -> "Бригада грузчиков". Модели семейства e5 ждут префиксы
"query: " у запроса и "passage: " у документа.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..features.filters import parse_filter

_SERVICE_RE = re.compile(r"(?:Услуга|Название услуги) (.+?)(?= Стоимость| Продолжительность| Начальная цена| Тип стоимости| Площадь|$)")


def _known_values(filters) -> dict[str, list[str]]:
    """Значения "Вид услуги" и "Тип услуги", которые встречаются в фильтрах поиска."""
    vals = {"Вид услуги": set(), "Тип услуги": set()}
    for f in filters:
        for k, vs in parse_filter(f).items():
            if k in vals:
                vals[k].update(vs)
    return {k: sorted(v, key=len, reverse=True) for k, v in vals.items()}


def build_passages(items: pd.DataFrame, filters, desc_chars: int = 300) -> list[str]:
    """Текст объявления для модели: заголовок, вид/тип услуги, названия услуг
    из прайс-листа и начало описания. Остальные параметры ("Рабочие дни...",
    "График работы...") для модели просто шум, их не беру."""
    known = _known_values(filters)
    vid_re = re.compile("Вид услуги (" + "|".join(map(re.escape, known["Вид услуги"])) + ")")
    tip_re = re.compile("Тип услуги (" + "|".join(map(re.escape, known["Тип услуги"])) + ")")
    out = []
    for title, params, desc in zip(items["item_title_raw"].values, items["item_infm_params_text"].values,
                                   items["item_description_raw"].values):
        parts = [str(title).strip()]
        m1, m2 = vid_re.search(params), tip_re.search(params)
        cat = ", ".join(x.group(1) for x in (m1, m2) if x)
        if cat:
            parts.append(cat)
        services = []
        for s in _SERVICE_RE.findall(params):
            s = s.strip()
            if s and s != "Своя услуга" and s not in services:
                services.append(s)
            if len(services) >= 6:
                break
        if services:
            parts.append("Услуги: " + "; ".join(services))
        d = " ".join(str(desc).split())[:desc_chars]
        if d:
            parts.append(d)
        out.append(". ".join(parts))
    return out


def load_st_model(name_or_path: str, device: str = "cuda", max_seq_length: int = 128):
    """Загрузка sentence-transformers модели.

    deepvk/USER-base (участвовала в сравнении моделей) не грузится в
    sentence-transformers 6.x из-за конфига модуля Normalize, поэтому для неё собираю
    ту же архитектуру вручную: трансформер -> mean pooling -> нормализация.
    Префикс по умолчанию отключаю, префиксы ставлю сам.
    """
    from sentence_transformers import SentenceTransformer, models
    try:
        model = SentenceTransformer(name_or_path, device=device)
    except TypeError:
        tr = models.Transformer(name_or_path, max_seq_length=max_seq_length)
        pool = models.Pooling(tr.get_word_embedding_dimension(), pooling_mode="mean")
        model = SentenceTransformer(modules=[tr, pool, models.Normalize()], device=device)
    model.max_seq_length = max_seq_length
    model.default_prompt_name = None
    return model


class DenseEncoder:
    def __init__(self, model_name_or_path: str, max_seq_length: int = 128, device: str = "cuda",
                 query_prefix: str = "query: ", passage_prefix: str = "passage: "):
        self.model = load_st_model(model_name_or_path, device, max_seq_length)
        if device == "cuda":
            self.model.half()          # fp16 на инференсе: вдвое быстрее, качество то же
        self.qp, self.pp = query_prefix, passage_prefix

    def encode(self, texts, is_query: bool, batch_size: int = 256) -> np.ndarray:
        pref = self.qp if is_query else self.pp
        # prompt="", чтобы библиотека не дописала префикс модели второй раз
        emb = self.model.encode([pref + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False, prompt="")
        return emb.astype(np.float16)
