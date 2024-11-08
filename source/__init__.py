"""
Configure the project.
"""

import os
import sys
import socket
import logging
import warnings
from pathlib import Path

# Configure the workspace
workspace = Path("/data/group_data/cx_group/scope")
workspace.mkdir(mode=0o770, parents=True, exist_ok=True)
os.environ["HF_HOME"] = Path(workspace, "hfhome").as_posix()

# Configure the logger
logger = logging.getLogger("scope")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Configure the warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Report the environment
logger.info("Hostname  : %s", socket.gethostname())
logger.info("Workspace : %s", workspace)
logger.info("Command   : %s", " ".join(sys.argv))
