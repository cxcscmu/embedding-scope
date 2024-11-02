"""
Implementation of openbmb/MiniCPM-Embedding.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.typing import NDArray
from transformers import AutoModel, AutoTokenizer
from source.interface import TextEmbedding


class MiniCPM(TextEmbedding):
    """
    Implementation of openbmb/MiniCPM-Embedding.

    References:
        https://huggingface.co/openbmb/MiniCPM-Embedding
    """

    name = "MiniCPM"
    size = 2304

    def __init__(self, devices: List[int]):
        assert len(devices) > 0
        self.devices = devices
        name = "openbmb/MiniCPM-Embedding"
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(
            name,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        )
        model = model.eval().to(devices[0])
        self.model = nn.DataParallel(model, devices)

    @torch.no_grad()
    def forward(self, texts: List[str]) -> NDArray[np.float32]:
        kwargs = {
            "max_length": 512,
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }
        encoded = self.tokenizer(texts, **kwargs)
        encoded = encoded.to(self.devices[0])
        outputs = self.model(**encoded)
        attention_mask = outputs.attention_mask
        last_hidden_state = outputs.last_hidden_state
        s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        embeddings = F.normalize(s / d, p=2, dim=1)
        return embeddings.detach().cpu().numpy()
