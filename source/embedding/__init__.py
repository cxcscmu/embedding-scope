"""
Implementation for the embedding interface.
"""

tokenizerKwargs = {
    "padding": True,
    "max_length": 512,
    "truncation": True,
    "return_tensors": "pt",
}

from .miniCPM import MiniCPM
from .bgeBase import BgeBase
