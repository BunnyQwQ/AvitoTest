"""Мелкие помощники: лог со временем и кэширование промежуточных результатов."""
import pickle
import time

import numpy as np

from . import config as C


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def cached_pickle(name: str, fn):
    p = C.CACHE_DIR / name
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    obj = fn()
    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return obj


def cached_npy(name: str, fn) -> np.ndarray:
    p = C.CACHE_DIR / name
    if p.exists():
        return np.load(p)
    arr = fn()
    np.save(p, arr)
    return arr
