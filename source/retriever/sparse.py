"""
The sparse retriever.

This file implements the sparse retriever. The similarity between two sparse
vectors is calculated using the dot product. The retriever supports batch
indexing and querying for efficiency.

The Elasticsearch server must be installed properly on the local machine
before using this retriever. In particular, `elasticsearch` must be available
in the system PATH.

@author: Hao Kang <haok@andrew.cmu.edu>
@date: December 15, 2024
"""

import os
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from source import hostname
from source.retriever import workspace


class Retriever:
    """
    The sparse retriever.

    This retriever is designed for sparse vectors, where each vector is a
    dictionary of feature values. The similarity between two vectors is
    calculated using the dot product.

    The retriever uses Elasticsearch as the backend for indexing and querying
    the vectors. The Elasticsearch server is started when entering the context
    and terminated when exiting the context to avoid resource leakage. It must
    be installed properly on the local machine before using this retriever. In
    particular, `elasticsearch` must be available in the system PATH.

    The retriever supports batch indexing and querying for efficiency. By
    default, the retriever uses all available CPU cores for parallel
    processing, which is determined by the SLURM environment variable, or the
    number of CPU cores otherwise.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the sparse retriever.

        Parameters
        ----------
        name : str
            The name of the retriever. It is used to isolate the workspace
            from other retrievers and to identify the Elasticsearch index.
        """
        self.name = name.lower()
        self.ncpu = os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())
        self.workspace = Path(workspace, name, "sparse")
        shutil.rmtree(self.workspace, ignore_errors=True)
        self.workspace.mkdir(mode=0o770, parents=True)

    def _run_server(self) -> subprocess.Popen:
        """
        Run the Elasticsearch server.

        The server is started in the background and the standard output and
        error are redirected to /dev/null. The server is ready when the ping
        method of the Elasticsearch client returns True.

        We assume the compute nodes are secure and reliable, so we disable the
        security features of Elasticsearch for simplicity.

        Returns
        -------
        subprocess.Popen
            The Elasticsearch server process.
        """
        args = [
            "elasticsearch",
            f"-Enode.name={hostname}",
            f"-Epath.data={self.workspace}",
            f"-Epath.logs={self.workspace}",
            f"-Expack.security.enabled=false",
            f"-Ehttp.port=9200",
        ]
        return subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _new_client(self) -> Elasticsearch:
        """
        Create a new Elasticsearch client.

        This method is blocking until the server is ready.

        Returns
        -------
        Elasticsearch
            A new Elasticsearch client.
        """
        client = Elasticsearch(
            hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}],
            timeout=60 * 30,
        )
        while not client.ping():
            if self.server.poll() is not None:
                raise RuntimeError("Elasticsearch server is not running.")
            time.sleep(1)
        client.indices.create(
            index="sparse",
            mappings={
                "properties": {
                    "features": {
                        "type": "sparse_vector",
                    }
                },
            },
        )
        return client

    def __enter__(self):
        """
        Start the Elasticsearch server and create a new client.
        """
        self.server = self._run_server()
        self.client = self._new_client()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Close the client and terminate the server.
        """
        self.client.close()
        self.server.terminate()
        self.server.wait()

    def batch_index(self, payload: Dict[str, Dict[str, float]]) -> None:
        """
        Index a batch of sparse vectors.

        Parameters
        ----------
        payload : Dict[str, Dict[str, float]]
            A dictionary of sparse vectors, where the key is the document ID
            and the value is a dictionary of feature values.
        """
        bulk(
            self.client,
            [
                {
                    "_index": "sparse",
                    "_id": pid,
                    "_source": {"features": features},
                }
                for pid, features in payload.items()
            ],
        )
        self.client.indices.refresh(index="sparse")

    def batch_query(
        self, payload: List[Dict[str, float]], top_k: int
    ) -> List[List[str]]:
        """
        Query a batch of sparse vectors.

        The similarity between two sparse vectors is calculated using the dot
        product. See the references for more details.

        Parameters
        ----------
        payload : List[Dict[str, float]]
            A list of sparse vectors, where each element is a dictionary of
            feature values.
        top_k : int
            The number of most similar documents to return.

        Returns
        -------
        List[List[str]]
            A list of lists of document IDs, where each inner list contains
            the IDs of the most similar documents for the corresponding query.

        References
        ----------
        - https://www.elastic.co/guide/en/elasticsearch/reference/current/sparse-vector.html
        - https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-sparse-vector-query.html
        """
        batch = []
        for features in payload:
            batch.append({"index": "sparse"})
            batch.append(
                {
                    "query": {
                        "sparse_vector": {
                            "field": "features",
                            "query_vector": features,
                        }
                    },
                    "size": top_k,
                }
            )
        response = self.client.msearch(
            body=batch,
            max_concurrent_searches=self.ncpu,
        )
        indices = [
            [x["_id"] for x in result["hits"]["hits"]]
            for result in response["responses"]
        ]
        scores = [
            [x["_score"] for x in result["hits"]["hits"]]
            for result in response["responses"]
        ]
        return indices, scores
