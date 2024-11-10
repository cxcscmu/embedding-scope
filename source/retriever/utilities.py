"""
Utilities for retriever.
"""

import os
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List
from source import logger


def evaluateRetrieval(
    relevance: Dict[str, Dict[str, int]],
    retrieved: Dict[str, List[str]],
    evaluated: Path,
) -> None:
    """
    Evaluate the retrieval performance.

    :param relevance: The relevance scores for each query.
    :param retrieved: The retrieved passages for each query.
    :param evaluated: The file to write the evaluation results.
    """
    with NamedTemporaryFile(mode="w") as qrel, NamedTemporaryFile(mode="w") as qret:
        logger.info("Writing relevance and retrieved files.")
        for queryID, results in relevance.items():
            for passageID, score in results.items():
                qrel.write(f"{queryID} 0 {passageID} {score}\n")
        for queryID, results in retrieved.items():
            for rank, passageID in enumerate(results):
                qret.write(f"{queryID} 0 {passageID} 0 {-rank} 0\n")
        qrel.flush()
        os.fsync(qrel.fileno())
        qret.flush()
        os.fsync(qret.fileno())

        logger.info("Evaluating the retrieval performance.")
        args = ["trec_eval", "-m", "all_trec", qrel.name, qret.name]
        with evaluated.open("w") as out:
            subprocess.run(args, stdout=out, check=True)
