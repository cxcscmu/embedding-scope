"""
The trainer module.
"""

from pathlib import Path
from source import workspace

workspace = Path(workspace, "trainer.1201")
workspace.mkdir(mode=0o770, parents=True, exist_ok=True)
