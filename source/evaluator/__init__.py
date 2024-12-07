"""Evaluate various performance metrics for the autoencoder.

This module contains functions to evaluate the performance of the autoencoder
using various metrics. These metrics include the mean-squared error (MSE), the
Kullback-Leibler divergence (KLD) on the query-document distribution, the
retrieval performance on the reconstructed embeddings, and the retrieval
performance on the sparse latent features.

Note that the evaluator module relies on the retriever module to perform the
information retrieval and trec_eval to evaluate the retrieval results. You may
refer to the retriever module for the implementation details.
"""
