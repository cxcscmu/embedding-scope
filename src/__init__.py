"""
@brief: Configure the project workspace.
@author: Hao Kang <haok@andrew.cmu.edu>
"""

from pathlib import Path

workspace = Path("/data/group_data/cx_group/scope")
workspace.mkdir(mode=0o770, parents=True, exist_ok=True)
