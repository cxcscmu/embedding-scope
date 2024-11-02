"""
Configure the project environment.
"""

import os
import sys
import socket
from pathlib import Path
from rich.console import Console

workspace = Path("/data/group_data/cx_group/scope")
workspace.mkdir(mode=0o770, parents=True, exist_ok=True)
os.environ["HF_HOME"] = Path(workspace, "hfhome").as_posix()

console = Console(width=120, log_path=False)
console.log(f"Hostname  : {socket.gethostname()}")
console.log(f"Workspace : {workspace}")
console.log(f"Command   : {' '.join(sys.argv)}")
