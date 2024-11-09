"""
The text retrieval dataset.
"""

from pathlib import Path
from source import workspace

workspace = Path(workspace, "dataset/textRetrieval")
workspace.mkdir(mode=0o770, parents=True, exist_ok=True)

from .msMarco import MsMarcoDataset
