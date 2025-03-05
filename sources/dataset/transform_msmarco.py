"""
@brief: Transform the MS MARCO dataset.
@author: Hao Kang <haok@andrew.cmu.edu>
"""

import os
import pickle
from pathlib import Path
from functools import partial
from typing import Dict
from sources import logger

workspace = Path(os.environ["DATASET_DIR"], "msmarco/text")
workspace.mkdir(mode=0o770, parents=True, exist_ok=True)


def convert_split_passages():
    i, data = 0, []
    path1 = Path(workspace, "collection.tsv")
    logger.info(f"Converting {path1}")
    with open(path1, 'r') as file1:
        for line in file1:
            _id, text = line.split("\t")
            if text == "": continue
            data.append((_id, text))
            if len(data) == 1_000_000:
                path2 = path1.with_name(f"passages-{i:04d}.bin")
                with open(path2, 'wb') as file2:
                    pickle.dump(data, file2)
                i += 1
                data.clear()
    if data:
        path2 = path1.with_name(f"passages-{i:04d}.bin")
        with open(path2, 'wb') as file2:
            pickle.dump(data, file2)
    path1.unlink()


def convert_queries(split: str):
    data = []
    path1 = Path(workspace, f"queries.{split}.tsv")
    logger.info(f"Converting {path1}")
    with open(path1, 'r') as file1:
        for line in file1:
            _id, text = line.split("\t")
            if text == "": continue
            data.append((_id, text))
    path2 = path1.with_name(f"queries.{split}.bin")
    with open(path2, 'wb') as file2:
        pickle.dump(data, file2)
    path1.unlink()


def convert_qrels(split: str):
    data: Dict[str, Dict[str, int]] = dict()
    path1 = Path(workspace, f"qrels.{split}.tsv")
    logger.info(f"Converting {path1}")
    with open(path1, 'r') as file1:
        for line in file1:
            qid, _, pid, rel = line.split("\t")
            rel = int(rel)
            if qid not in data:
                data[qid] = dict()
            data[qid][pid] = rel
    path2 = path1.with_name(f"qrels.{split}.bin")
    with open(path2, 'wb') as file2:
        pickle.dump(data, file2)
    path1.unlink()


def main():
    procid = int(os.getenv("SLURM_PROCID", "0"))
    ntasks = int(os.getenv("SLURM_NTASKS", "1"))
    for i, fn in enumerate([
        convert_split_passages, partial(convert_queries, "train"),
        partial(convert_queries, "dev"), partial(convert_queries, "eval"),
        partial(convert_qrels, "train"), partial(convert_qrels, "dev")
    ]):
        if i % ntasks == procid: fn()


if __name__ == '__main__':
    main()
