"""
The evaluator module.

This module implements various pipelines for evaluating the performance of the
sparse autoencoder after training. These evaluation metrics include the mean
squared error (MSE) for point-wise reconstruction, the Kullback-Leibler (KL)
divergence (KLD) on the query-document distribution, the retrieval performance
on the reconstructed embeddings and the sparse latent features.

@author: Hao Kang <haok@andrew.cmu.edu>
@date: December 15, 2024
"""

from pathlib import Path

# We use a different workspace than the one speicified in the source module
# since the evaluation results are "critical" and we would like to ensure
# proper backup. The local directories mounted on the login nodes are backed
# up regularly unlike the `/data` workspace.
workspace = Path("logging")
workspace.mkdir(mode=0o770, exist_ok=True)
