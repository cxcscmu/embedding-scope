"""
Implementation of BAAI/bge-base-en-v1.5.
"""

from typing import List
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from src.interface import TextEmbedding


class BgeBase(TextEmbedding):
    """
    Implementation of BAAI/bge-base-en-v1.5.
    """

    name = "BgeBase"
    size = 768
    type = np.float32

    def __init__(self, devices: List[int]):
        assert len(devices) > 0
        self.devices = devices
        name = "BAAI/bge-base-en-v1.5"
        self.tokenizer = BertTokenizerFast.from_pretrained(name)
        assert isinstance(self.tokenizer, BertTokenizerFast)
        model = BertModel.from_pretrained(name)
        assert isinstance(model, BertModel)
        model = model.eval().to(devices[0])
        self.model = nn.DataParallel(model, devices)

    def forward(self, documents: List[str]) -> NDArray[np.float32 | np.float16]:
        kwargs = {"padding": True, "truncation": True, "return_tensors": "pt"}
        encoded = self.tokenizer(documents, **kwargs)
        outputs = self.model.forward(**encoded.to(self.devices[0]))
        assert isinstance(outputs, BaseModelOutputWithPoolingAndCrossAttentions)
        hiddens = outputs.last_hidden_state
        hiddens = F.normalize(hiddens[:, 0], p=2, dim=1)
        return hiddens.detach().cpu().numpy()
