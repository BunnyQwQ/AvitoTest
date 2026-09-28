"""Токенизация и лемматизация (pymorphy3).

pymorphy разбирает слово медленно, поэтому каждое уникальное слово разбираю один
раз и держу словарь слово -> лемма в памяти (и на диске, если вызвать save_lemma_cache).
"""
from __future__ import annotations

import pickle
import re

from . import config as C

TOKEN_RE = re.compile(r"[a-zа-я0-9]+")

# Только служебные слова. "Услуги", "ремонт" и т.п. не выкидываю: у них и так
# низкий IDF, а в коротком запросе вроде "услуги бухгалтера" они что-то значат.
STOPWORDS = set("""
в во на по с со к ко о об обо от до для из за у и или а но не ни без при под над
через что как это мне меня нам нас вам вас их его ее её же ли бы то все всё весь
этот эта эти тот та те который которая которые где когда там тут очень есть быть
""".split())

_LEMMA_CACHE_PATH = C.CACHE_DIR / "lemma_cache.pkl"
_lemma_cache: dict[str, str] | None = None
_morph = None


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower().replace("ё", "е"))


def _get_morph():
    global _morph
    if _morph is None:
        import pymorphy3
        _morph = pymorphy3.MorphAnalyzer()
    return _morph


def _load_cache() -> dict[str, str]:
    global _lemma_cache
    if _lemma_cache is None:
        if _LEMMA_CACHE_PATH.exists():
            with open(_LEMMA_CACHE_PATH, "rb") as f:
                _lemma_cache = pickle.load(f)
        else:
            _lemma_cache = {}
    return _lemma_cache


def save_lemma_cache() -> None:
    with open(_LEMMA_CACHE_PATH, "wb") as f:
        pickle.dump(_load_cache(), f, protocol=pickle.HIGHEST_PROTOCOL)


def lemmatize_vocab(words) -> dict[str, str]:
    cache = _load_cache()
    morph = _get_morph()
    for w in words:
        if w in cache:
            continue
        # латиницу и числа оставляю как есть, для кириллицы беру самый вероятный разбор
        if w.isascii():
            cache[w] = w
        else:
            cache[w] = morph.parse(w)[0].normal_form.replace("ё", "е")
    return cache


def lemmas(text: str, drop_stop: bool = False) -> list[str]:
    cache = _load_cache()
    toks = tokenize(text)
    missing = [t for t in toks if t not in cache]
    if missing:
        lemmatize_vocab(missing)
    out = [cache[t] for t in toks]
    if drop_stop:
        out = [t for t in out if t not in STOPWORDS]
    return out
