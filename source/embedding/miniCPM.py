"""
Implementation of openbmb/MiniCPM-Embedding.
"""

from typing import List
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
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
        batch_dict = self.tokenizer(texts, **kwargs).to(self.devices[0])
        outputs = self.model(**batch_dict)
        attention_mask = batch_dict["attention_mask"]
        hidden = outputs.last_hidden_state
        s = torch.sum(hidden * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        embeddings = F.normalize(s / d, p=2, dim=1)
        return embeddings.detach().cpu().numpy()
