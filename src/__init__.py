"""
Configure the project workspace.
"""

import os
from pathlib import Path

workspace = Path("/data/group_data/cx_group/scope")
workspace.mkdir(mode=0o770, parents=True, exist_ok=True)
os.environ["HF_HOME"] = Path(workspace, "hfhome").as_posix()
