"""
Initialize the project.
"""

import os
import sys
import socket
import logging
import warnings
from pathlib import Path

# Setup the workspace
workspace = Path("/data/group_data/cx_group/scope")
workspace.mkdir(mode=0o770, parents=True, exist_ok=True)
os.environ["HF_HOME"] = Path(workspace, "hfhome").as_posix()

# Setup the logger
logger = logging.getLogger("scope")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(fmt="%(asctime)s | %(message)s")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Suppress the warnings
warnings.filterwarnings("ignore")

# Report the environment
hostname = socket.gethostname()
logger.info("Hostname  : %s", hostname)
logger.info("Workspace : %s", workspace)
logger.info("Command   : %s", " ".join(sys.argv))
