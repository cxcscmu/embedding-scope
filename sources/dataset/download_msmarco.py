"""
@brief: Download the MS MARCO dataset.
@author: Hao Kang <haok@andrew.cmu.edu>
"""

import os
import shutil
import subprocess
import requests
from pathlib import Path
from sources import logger

workspace = Path(os.environ["DATASET_DIR"], "msmarco/text")
workspace.mkdir(mode=0o770, parents=True, exist_ok=True)
prefix = "https://msmarco.z22.web.core.windows.net/msmarcoranking"


def retrieve(link: str, path: Path):
    with requests.get(link, stream=True) as r:
        with open(path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)


def download_passages():
    link = prefix + "/collection.tar.gz"
    path = Path(workspace, "collection.tar.gz")
    logger.info(f"Downloading {link}")
    retrieve(link, path)
    logger.info(f"Extracting {path}")
    subprocess.run(["tar", "-xzvf", "collection.tar.gz"], cwd=workspace, check=True)
    path.unlink()


def download_queries():
    link = prefix + "/queries.tar.gz"
    path = Path(workspace, "queries.tar.gz")
    logger.info(f"Downloading {link}")
    retrieve(link, path)
    logger.info(f"Extracting {path}")
    subprocess.run(["tar", "-xzvf", "queries.tar.gz"], cwd=workspace, check=True)
    path.unlink()


def download_train_relevance():
    link = prefix + "/qrels.train.tsv"
    path = Path(workspace, "qrels.train.tsv")
    logger.info(f"Downloading {link}")
    retrieve(link, path)


def download_dev_relevance():
    link = prefix + "/qrels.dev.tsv"
    path = Path(workspace, "qrels.dev.tsv")
    logger.info(f"Downloading {link}")
    retrieve(link, path)


def main():
    procid = int(os.getenv("SLURM_PROCID", "0"))
    ntasks = int(os.getenv("SLURM_NTASKS", "1"))
    for i, fn in enumerate([
        download_passages, download_queries, 
        download_train_relevance, download_dev_relevance
    ]):
        if i % ntasks == procid: fn()


if __name__ == '__main__':
    main()
