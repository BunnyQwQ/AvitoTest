"""Полный пайплайн: parquet-файлы из data/ -> submission/answer.csv.

    python main.py

1. Данные и валидация: 8 «псевдо-бенчмарков» из train, устроенных как настоящий.
2. Индексы по корпусу: BM25F, символьные n-граммы, эмбеддинги дообученной e5.
3. Кандидаты и признаки для валидационных порций (история = train без валидации).
4. Два ранкера LightGBM: основной и «без рубрикатора» для поиска по всем категориям.
   Учатся на порциях 0–4, ранняя остановка на 5, оценка на 6–7, потом переобучаются на всех.
5. Бенчмарк: история = весь train, кандидаты, ранжирование, топ-50 -> answer.csv.
Промежуточные результаты кэшируются в cache/, повторный запуск пропускает готовое.
"""
import json
import os
import time

import numpy as np
import pandas as pd

from src import config as C
from src import ranker as RK
from src.candidates import GenConfig, Resources, generate
from src.data import all_filters, build_all
from src.features.filters import FilterMatcher
from src.features.history import HistoryStats
from src.features.locations import LocationModel
from src.features.query_clf import QueryAttrClassifier
from src.metrics import recall_at_k
from src.retrieval.bm25 import BM25F
from src.retrieval.chargram import CharIndex
from src.retrieval.dense import DenseEncoder, build_passages
from src.retrieval.finetune import train_encoder
from src.submission import write_answer
from src.utils import cached_npy, cached_pickle, log


def main():
    # детерминированный cuBLAS: эмбеддинги на GPU считаются одинаково от запуска к запуску
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    t0 = time.time()
    cfg = GenConfig()

    # ---------- 1. данные ----------
    d = build_all()
    items, history, train, vq, vqi, bq = d["items"], d["history"], d["train_keys"], d["vq"], d["vqi"], d["bq"]
    filters = all_filters(d)
    item_ids = items["item_id"].to_numpy()
    id2idx = {x: i for i, x in enumerate(item_ids)}
    in_bench = items["in_bench"].to_numpy()
    log(f"данные: объявлений {len(items)}, история {len(history)}, валидация {len(vq)}, бенчмарк {len(bq)}")

    # ---------- 2. индексы и эмбеддинги ----------
    bm25 = cached_pickle("bm25.pkl", lambda: BM25F(items, weights=(5, 1, 1), bs=(0.3, 0.75, 0.75), k1=0.8))
    char = cached_pickle("char.pkl", lambda: CharIndex(items))
    fm = FilterMatcher(items)
    if not C.MODEL_DIR.exists():
        train_encoder(d)
    enc = None

    def get_enc():
        nonlocal enc
        if enc is None:
            enc = DenseEncoder(str(C.MODEL_DIR))
        return enc
    item_emb = cached_npy("emb_items.npy", lambda: get_enc().encode(build_passages(items, filters), is_query=False))
    vq_emb = cached_npy("emb_vq.npy", lambda: get_enc().encode(vq["qn"].tolist(), is_query=True))
    bq_emb = cached_npy("emb_bq.npy", lambda: get_enc().encode(bq["qn"].tolist(), is_query=True))
    del enc
    log("индексы и эмбеддинги готовы")

    # ---------- 3. кандидаты для валидационных порций ----------
    gold = vqi.groupby("query_id")["item_id"].apply(set).to_dict()
    chunk_paths = [C.CACHE_DIR / f"cand_val_{c}.parquet" for c in range(C.N_VAL_CHUNKS)]
    if not all(p.exists() for p in chunk_paths):
        clf_a = cached_pickle("clf_history.pkl", lambda: QueryAttrClassifier(max_iter=100).fit(history))
        R = Resources(items=items, bm25=bm25, char=char, loc=LocationModel(history, items), fm=fm, clf=clf_a,
                      hist=HistoryStats(history, items), item_emb=item_emb)
        P_val = clf_a.predict_proba(vq["qn"].tolist())
        for c, path in enumerate(chunk_paths):
            if path.exists():
                continue
            m = (vq["chunk"] == c).to_numpy()
            # корпус порции = объявления бенчмарка + правильные ответы этой порции
            allowed = in_bench.copy()
            allowed[[id2idx[x] for x in vqi.loc[vqi["query_id"].isin(set(vq.loc[m, "query_id"])), "item_id"]]] = True
            df = generate(vq[m], {c: allowed}, R, vq_emb[m], P_val[m], gold, cfg)
            df["chunk"] = np.int8(c)
            df.to_parquet(path, index=False)
            log(f"порция {c}: {len(df)} строк, {df.groupby('query_id').size().mean():.0f} кандидатов на запрос")
        del R
    cand = pd.concat([pd.read_parquet(p) for p in chunk_paths], ignore_index=True)

    # ---------- 4. ранкеры ----------
    tr = cand[cand["chunk"].isin(C.TRAIN_CHUNKS)]
    va = cand[cand["chunk"] == C.EARLY_STOP_CHUNK]
    ho = cand[cand["chunk"].isin(C.HOLDOUT_CHUNKS)]
    hold_q = vq.loc[vq["chunk"].isin(C.HOLDOUT_CHUNKS)]
    hold_gold = {q: gold[q] for q in hold_q["query_id"]}
    report = {"holdout_queries": len(hold_q)}
    pool = {q: item_ids[g.to_numpy()].tolist() for q, g in ho.groupby("query_id")["item_idx"]}
    report["pool_recall"] = recall_at_k(pool, hold_gold, 10 ** 6)
    models = {}
    for name, exclude in [("main", ()), ("all_categories", RK.CATEGORY_SPECIFIC)]:
        model, feats = RK.train(tr, va, exclude=exclude)
        pred = RK.predict_topk(model, feats, ho, item_ids)
        report[f"recall@50_{name}"] = recall_at_k(pred, hold_gold, 50)
        if name == "main":
            seen = set(history["qn"])
            for seg, mask in [("filter", hold_q["flt"] != ""), ("no_filter", hold_q["flt"] == ""),
                              ("seen_text", hold_q["qn"].isin(seen)), ("new_text", ~hold_q["qn"].isin(seen))]:
                report[f"recall@50_main_{seg}"] = recall_at_k(pred, {q: gold[q] for q in hold_q.loc[mask, "query_id"]}, 50)
        report[f"best_iter_{name}"] = int(model.best_iteration)
        # финальная модель — на всех 8 порциях, деревьев на 10% больше
        final, feats = RK.train(cand, None, num_boost_round=int(model.best_iteration * 1.1), exclude=exclude)
        final.save_model(str(C.CACHE_DIR / f"ranker_{name}.txt"))
        models[name] = (final, feats)
    log("валидация: " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in report.items()},
                                   ensure_ascii=False))
    with open(C.CACHE_DIR / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    del cand, tr, va, ho

    # ---------- 5. бенчмарк ----------
    clf_b = cached_pickle("clf_full.pkl", lambda: QueryAttrClassifier(max_iter=100).fit(train))
    R = Resources(items=items, bm25=bm25, char=char, loc=LocationModel(train, items), fm=fm, clf=clf_b,
                  hist=HistoryStats(train, items), item_emb=item_emb)
    cand_b = generate(bq.assign(chunk=-1), {-1: in_bench}, R, bq_emb, clf_b.predict_proba(bq["qn"].tolist()),
                      None, cfg)
    cand_b.to_parquet(C.CACHE_DIR / "cand_bench.parquet", index=False)
    # запросы «по всем категориям» ранжирует модель без признаков рубрикатора Услуг
    cat0 = set(bq.loc[bq["search_category"] == 0, "query_id"])
    pred_b = RK.predict_topk(*models["main"], cand_b[~cand_b["query_id"].isin(cat0)], item_ids)
    pred_b.update(RK.predict_topk(*models["all_categories"], cand_b[cand_b["query_id"].isin(cat0)], item_ids))
    write_answer(bq, pred_b, set(item_ids[in_bench]))
    log(f"готово за {(time.time() - t0) / 60:.1f} мин")


if __name__ == "__main__":
    main()
