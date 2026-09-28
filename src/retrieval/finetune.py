"""Дообучение multilingual-e5-base на парах (текст запроса, выбранное объявление).

Функция потерь - MultipleNegativesRankingLoss: для каждой пары остальные объявления
из батча служат негативами. Cached-вариант (GradCache) позволяет батч 256 на 8 ГБ.
Обучаю только на history (train без валидационных запросов), иначе модель
запомнит ответы валидации и оценка будет завышена.
"""
from __future__ import annotations

import time

import pandas as pd

from .. import config as C
from ..data import all_filters
from .dense import build_passages, load_st_model


def make_pairs(history: pd.DataFrame, item_texts: pd.DataFrame, filters, max_per_text: int = 50) -> pd.DataFrame:
    """Уникальные пары (запрос, текст объявления). Частые запросы вроде "маникюр"
    обрезаю до 50 пар, чтобы модель не перекосило в сторону головы распределения:
    бенчмарк в основном из редких формулировок."""
    pairs = history[["qn", "item_id"]].drop_duplicates()
    pairs = pairs.sample(frac=1.0, random_state=C.SEED).groupby("qn").head(max_per_text)
    it = item_texts.drop_duplicates("item_id").set_index("item_id").loc[pairs["item_id"].unique()].reset_index()
    passages = dict(zip(it["item_id"], build_passages(it, filters)))
    pairs = pairs.assign(passage=pairs["item_id"].map(passages)).dropna()
    return pairs.sample(frac=1.0, random_state=C.SEED).reset_index(drop=True)


def finetune(base_model: str, pairs: pd.DataFrame, out_dir, epochs: int = 1, batch_size: int = 256,
             mini_batch: int = 64, lr: float = 3e-5, max_seq_length: int = 128):
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    from sentence_transformers.training_args import BatchSamplers

    torch.manual_seed(C.SEED)
    model = load_st_model(base_model, "cuda", max_seq_length)
    ds = Dataset.from_dict({"anchor": ("query: " + pairs["qn"]).tolist(),
                            "positive": ("passage: " + pairs["passage"]).tolist()})
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir) + "_ckpt", num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        learning_rate=lr, warmup_ratio=0.05, fp16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=100, save_strategy="no", seed=C.SEED, data_seed=C.SEED, report_to="none",
        dataloader_drop_last=True)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss)
    trainer.train()
    model.save(str(out_dir))
    return out_dir


def train_encoder(d: dict, out_dir=C.MODEL_DIR, base_model: str = C.BASE_MODEL):
    """Полный цикл дообучения: пары из истории -> модель в out_dir (~70 минут на RTX 3080 Laptop)."""
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    # в history нет описаний, поэтому тексты объявлений беру из исходного train
    tr_items = pd.read_parquet(C.TRAIN_PATH, columns=["item_id", "item_title_raw", "item_infm_params_text",
                                                        "item_description_raw"]).drop_duplicates("item_id")
    for c in ["item_title_raw", "item_infm_params_text", "item_description_raw"]:
        tr_items[c] = tr_items[c].fillna("")
    pairs = make_pairs(d["history"], tr_items, all_filters(d))
    print(f"пар для обучения: {len(pairs)}, уникальных текстов: {pairs['qn'].nunique()}", flush=True)
    t = time.time()
    finetune(base_model, pairs, out_dir)
    print(f"дообучение заняло {(time.time() - t) / 60:.1f} мин -> {out_dir}", flush=True)
