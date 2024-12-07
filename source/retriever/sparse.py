"""Retriever for sparse features.

This module implements the retriever for sparse features. The retriever relies
on the Elasticsearch engine to perform the retrieval. We assume the compute
node is reliable, which means the use of multiple replicas is not necessary.
In addition, we assume the compute node is secure, which means the use of SSL
could be omitted. On the other hand, we assume in most scenarios, there may be
grid search over sparse features extracted from different hyper parameters,
which prioritizes the search over multiple indices at once. We've therefore
assigned each retriever a separate engine with a unique index to boost the
scalability of the retrieval process. This assumes slurm array job is used to
perform the grid search, which assigns each job to a different node.
"""

import requests
from source import logger, hostname
from source.utilities import tqdm


class Retriever:
    """
    Retriever for sparse features.
    """

    def __init__(self):
        self._download_engine()

    def _download_engine(self):
        """
        Download the Elasticsearch engine.

        This method downloads the Elasticsearch engine from the official
        website to the compute node. The engine is then installed and
        configured to run without SSL and multiple replicas. We assume the
        platform for the compute node is linux x86-64.
        """
        logger.info("Download the Elasticsearch engine.")

        with requests.get(
            "https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.16.1-linux-x86_64.tar.gz",
            stream=True,
            allow_redirects=True,
            timeout=30 * 60,
        ) as response:
            response.raise_for_status()


if __name__ == "__main__":
    Retriever()
    raise RuntimeError("This module should not be executed as a script.")
