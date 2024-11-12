"""
The controller module.
"""

from pathlib import Path
from source import workspace

workspace = Path(workspace, "controller")
workspace.mkdir(mode=0o770, parents=True, exist_ok=True)
