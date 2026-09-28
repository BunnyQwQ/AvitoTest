"""Запись answer.csv и проверка формата по требованиям задания."""
import pandas as pd

from . import config as C
from .utils import log


def write_answer(bq: pd.DataFrame, pred: dict, corpus_ids: set, path=C.SUBMISSION_PATH) -> None:
    rows = []
    for qid in bq["query_id"]:
        top = [x for x in pred.get(qid, []) if x in corpus_ids]
        top = list(dict.fromkeys(top))[:C.TOP_K]
        rows.append((qid, " ".join(top)))
    path.parent.mkdir(parents=True, exist_ok=True)
    # \r\n явно, а не по умолчанию для ОС: отправленный файл писался на Windows,
    # так ответ совпадает с ним байт в байт на любой системе
    pd.DataFrame(rows, columns=["query_id", "answer"]).to_csv(path, index=False, encoding="utf-8",
                                                              lineterminator="\r\n")

    # перечитываю как строки (как это сделает проверяющая система) и проверяю всё из условия
    chk = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(chk.columns) == ["query_id", "answer"], "колонки"
    assert len(chk) == len(bq) and set(chk["query_id"]) == set(bq["query_id"]) and chk["query_id"].is_unique, "query_id"
    assert chk["query_id"].str.len().eq(16).all(), "длина query_id"
    for a in chk["answer"]:
        ids = a.split(" ")
        assert 0 < len(ids) <= C.TOP_K and len(set(ids)) == len(ids), "число или повторы item_id"
        assert all(len(x) == 16 and x in corpus_ids for x in ids), "item_id не из корпуса"
    lens = chk["answer"].str.split(" ").str.len()
    log(f"{path.name} записан и проверен: {len(chk)} запросов, объявлений в строке: мин {lens.min()}, "
        f"среднее {lens.mean():.1f}")
