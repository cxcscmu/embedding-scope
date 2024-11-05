"""
Serve as an interface for the application.
"""

from source.interface.embedding import TextEmbedding
from source.interface.dataset import PartitionType, TextRetrievalDataset
from source.interface.retriever import DenseRetriever
from source.interface.autoencoder import AutoEncoder
from source.interface.trainer import Trainer
