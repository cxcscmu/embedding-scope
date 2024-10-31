"""
Implementation of BAAI/bge-base-en-v1.5.
"""

from typing import List
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from transformers import AutoModel, AutoTokenizer
from source.interface import TextEmbedding


class BgeBase(TextEmbedding):
    """
    Implementation of BAAI/bge-base-en-v1.5.

    References:
        https://huggingface.co/BAAI/bge-base-en-v1.5
    """

    name = "BgeBase"
    size = 768

    def __init__(self, devices: List[int]):
        assert len(devices) > 0
        self.devices = devices
        name = "BAAI/bge-base-en-v1.5"
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        model = model.eval().to(devices[0])
        self.model = nn.DataParallel(model, devices)

    def forward(self, texts: List[str]) -> NDArray[np.float32]:
        kwargs = {
            "max_length": 512,
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        encoded = self.tokenizer(texts, **kwargs)
        outputs = self.model.forward(**encoded.to(self.devices[0]))
        hiddens = outputs.last_hidden_state
        hiddens = F.normalize(hiddens[:, 0], p=2, dim=1)
        return hiddens.detach().cpu().numpy()
