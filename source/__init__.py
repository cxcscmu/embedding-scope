"""
Configure the project workspace.
"""

import os
import socket
from pathlib import Path
from rich.console import Console

workspace = Path("/data/group_data/cx_group/scope")
workspace.mkdir(mode=0o770, parents=True, exist_ok=True)
os.environ["HF_HOME"] = Path(workspace, "hfhome").as_posix()

console = Console(width=120, log_path=False)
