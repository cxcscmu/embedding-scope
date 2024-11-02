"""
Implementation for the dataset interface.
"""

from pathlib import Path
from source import workspace

workspace = Path(workspace, "dataset")
workspace.mkdir(mode=0o770, exist_ok=True)
