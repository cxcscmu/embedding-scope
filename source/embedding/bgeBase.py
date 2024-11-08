"""
Implementation of BAAI/bge-base-en-v1.5.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.typing import NDArray
from transformers import AutoModel, AutoTokenizer
from source.interface.embedding import TextEmbedding
from source.embedding import tokenizerKwargs


class BgeBase(TextEmbedding):
    """
    Implementation of BAAI/bge-base-en-v1.5.

    References:
        https://huggingface.co/BAAI/bge-base-en-v1.5
    """

    name = "bgeBase"
    size = 768

    def __init__(self, devices: List[int]):
        assert len(devices) > 0
        self.devices = devices
        name = "BAAI/bge-base-en-v1.5"
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        model = model.eval().to(devices[0])
        self.model = nn.DataParallel(model, devices)

    @torch.no_grad()
    def forward(self, texts: List[str]) -> NDArray[np.float32]:
        kwargs = tokenizerKwargs
        encoded = self.tokenizer(texts, **kwargs)
        encoded = encoded.to(self.devices[0])
        outputs = self.model.forward(**encoded)
        hiddens = outputs.last_hidden_state
        hiddens = F.normalize(hiddens[:, 0], p=2, dim=1)
        return hiddens.detach().cpu().numpy()
