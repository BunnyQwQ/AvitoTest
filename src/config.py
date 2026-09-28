"""Пути и константы проекта."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.parquet"
QUERIES_PATH = DATA_DIR / "benchmark_queries.parquet"
ITEMS_PATH = DATA_DIR / "benchmark_items.parquet"

# Промежуточные артефакты (индексы, эмбеддинги, кандидаты). Папку можно подменить,
# чтобы прогнать всё с нуля, не трогая основной кэш:  AVITO_CACHE_DIR=cache_check python main.py
CACHE_DIR = Path(os.environ.get("AVITO_CACHE_DIR", ROOT / "cache"))
CACHE_DIR.mkdir(exist_ok=True)

# Дообученный multilingual-e5-base. Если папки нет, main.py обучит модель сам (~70 минут на GPU).
MODEL_DIR = ROOT / "models" / "e5-avito"
BASE_MODEL = "intfloat/multilingual-e5-base"

SUBMISSION_PATH = Path(os.environ.get("AVITO_SUBMISSION", ROOT / "submission" / "answer.csv"))

SEED = 42
TOP_K = 50

# Валидация: 8 «псевдо-бенчмарков» из train по 2452 запроса — столько же, сколько в бенчмарке.
CHUNK_SIZE = 2452
N_VAL_CHUNKS = 8
TRAIN_CHUNKS = (0, 1, 2, 3, 4)   # обучение ранкера
EARLY_STOP_CHUNK = 5             # ранняя остановка
HOLDOUT_CHUNKS = (6, 7)          # честная оценка
