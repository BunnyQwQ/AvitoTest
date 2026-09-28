"""Загрузка данных и построение валидации.

Главное наблюдение из EDA: бенчмарк собран по принципу «один запрос на уникальный
текст» — доля текстов, которые встречаются в train, у бенчмарка 37.4%, а при таком
же отборе из train 37.8% (если брать случайные события поиска, было бы ~87%).
Поэтому валидационные запросы отбираю так же. Кроме того, в бенчмарке нет запросов
с фильтром «Автосервис» — их в валидацию не беру.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

ITEM_COLS = ["item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
             "item_category_id", "item_microcat_id", "item_price", "item_rating",
             "item_rating_reviews_count", "item_location_id", "item_latitude", "item_longitude",
             "item_is_phone_hidden", "item_is_message_forbidden"]


def norm_query(s: str) -> str:
    return " ".join(str(s).lower().replace("ё", "е").split())


def add_query_keys(df: pd.DataFrame) -> pd.DataFrame:
    """qn — нормализованный текст, flt — фильтр, qkey — «запрос целиком»
    (текст + локация + фильтр + категория). Строки train с одним qkey — разные
    выборы по одному и тому же запросу."""
    df["qn"] = df["search_query"].map(norm_query)
    df["flt"] = df["search_infm_params_text"].fillna("")
    df["qkey"] = (df["qn"] + "|" + df["search_location_id"].astype(str) + "|"
                  + df["flt"] + "|" + df["search_category"].astype(str))
    return df


def _fix_item_types(df: pd.DataFrame) -> pd.DataFrame:
    # цена и координаты лежат в parquet как decimal128
    for c in ["item_price", "item_latitude", "item_longitude"]:
        df[c] = df[c].astype("float64")
    for c in ["item_title_raw", "item_description_raw", "item_infm_params_text"]:
        df[c] = df[c].fillna("")
    return df


def load_train() -> pd.DataFrame:
    return add_query_keys(_fix_item_types(pd.read_parquet(C.TRAIN_PATH)))


def load_bench_queries() -> pd.DataFrame:
    return add_query_keys(pd.read_parquet(C.QUERIES_PATH))


def load_bench_items() -> pd.DataFrame:
    return _fix_item_types(pd.read_parquet(C.ITEMS_PATH))


def make_validation(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Псевдо-бенчмарки из train: vq — запросы (по одному на текст) с номером порции,
    vqi — их правильные ответы (query_id, item_id)."""
    keys = (train.groupby("qkey", sort=False)
            .agg(qn=("qn", "first"), flt=("flt", "first"),
                 search_query=("search_query", "first"),
                 search_location_id=("search_location_id", "first"),
                 search_is_delivery_search=("search_is_delivery_search", "first"),
                 search_infm_params_text=("search_infm_params_text", "first"),
                 search_category=("search_category", "first"))
            .reset_index())
    keys = keys[~keys["flt"].str.contains("Автосервис") & (keys["search_is_delivery_search"] == 0)]
    # равномерно по текстам: перемешиваю и оставляю по одному запросу на текст
    keys = keys.sample(frac=1.0, random_state=C.SEED).drop_duplicates("qn")
    n = C.CHUNK_SIZE * C.N_VAL_CHUNKS
    vq = keys.sample(n=n, random_state=C.SEED + 1).reset_index(drop=True)
    vq["chunk"] = np.arange(n) // C.CHUNK_SIZE
    vq["query_id"] = ["v" + str(i).zfill(15) for i in range(n)]
    vqi = (train.loc[train["qkey"].isin(set(vq["qkey"])), ["qkey", "item_id"]]
           .drop_duplicates()
           .merge(vq[["qkey", "query_id"]], on="qkey")[["query_id", "item_id"]])
    return vq, vqi


def all_filters(d: dict) -> list[str]:
    """Все строки фильтров из train и бенчмарка — из них берётся словарь значений «Вид/Тип услуги»."""
    return sorted(set(d["train_keys"]["flt"]) | set(d["bq"]["flt"]))


def build_all(force: bool = False) -> dict:
    """Готовит (и кэширует) все таблицы:
      items      — корпус бенчмарка + правильные ответы валидации из train (in_bench отмечает корпус);
      history    — train без валидационных запросов: на нём считается всё «историческое»;
      train_keys — весь train без описаний (для финального предсказания);
      vq, vqi, bq — валидационные запросы, их ответы и запросы бенчмарка.
    """
    paths = {k: C.CACHE_DIR / f"{k}.parquet" for k in ["items", "history", "vq", "vqi", "bq", "train_keys"]}
    if not force and all(p.exists() for p in paths.values()):
        return {k: pd.read_parquet(p) for k, p in paths.items()}

    train = load_train()
    bench_items = load_bench_items()
    bq = load_bench_queries()
    vq, vqi = make_validation(train)

    # признаки объявлений в train и в корпусе — один срез (совпадают на 100%),
    # поэтому ответы валидации можно просто подмешать в корпус
    tr_items = train.drop_duplicates("item_id")[ITEM_COLS]
    items = pd.concat([bench_items[ITEM_COLS].assign(in_bench=True),
                       tr_items[~tr_items["item_id"].isin(set(bench_items["item_id"]))].assign(in_bench=False)],
                      ignore_index=True)
    need = set(vqi["item_id"])
    items = items[items["in_bench"] | items["item_id"].isin(need)].reset_index(drop=True)

    val_keys = set(vq["qkey"])
    history = train[~train["qkey"].isin(val_keys)].drop(columns=["item_description_raw"]).reset_index(drop=True)
    train_keys = train.drop(columns=["item_description_raw"]).reset_index(drop=True)

    out = {"items": items, "history": history, "vq": vq, "vqi": vqi, "bq": bq, "train_keys": train_keys}
    for k, df in out.items():
        df.to_parquet(paths[k], index=False)
    return out
