"""NuRec clips whose annotations cannot be trusted for EPDMS scoring.

Some clips carry vehicle boxes that no camera frame supports: the box is tracked
across all nine samples of the scoring window, its position, heading, velocity
and size are mutually consistent, and yet the road at that spot is empty.  Every
geometric and kinematic test tried against them fails to separate them from real
traffic -- the only thing wrong is that nothing is there, and that lives in the
image, not in the boxes.  See ``docs/nurec/NC_HUMAN_FILTER.md`` section 9 for
the tests and why each was rejected.

Until an appearance- or LiDAR-based check exists, those clips are dropped whole.
The cost is small: 89 of 48,038 tokens (0.19%), removing 14 of the 971 frames
where the human trajectory scores zero on no_at_fault_collisions (1.4%).

The list is data, not code -- ``assets/nurec/excluded_clips.txt``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import FrozenSet, Optional

LIST_ENV = "NUREC_EXCLUDED_CLIPS"
DEFAULT_LIST = "assets/nurec/excluded_clips.txt"


def _list_path() -> Optional[Path]:
    override = os.environ.get(LIST_ENV)
    if override:
        return Path(override)
    root = os.environ.get("AXE_ROOT") or os.environ.get("NAVSIM_DEVKIT_ROOT")
    if not root:
        return None
    return Path(root) / DEFAULT_LIST


@lru_cache(maxsize=1)
def excluded_logs() -> FrozenSet[str]:
    """Log names to skip. Empty when the list is absent, so nothing breaks."""
    path = _list_path()
    if path is None or not path.is_file():
        return frozenset()
    names = set()
    for line in path.read_text().splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            names.add(name)
            names.add(name.removeprefix("nurec-"))
    return frozenset(names)


def is_excluded(log_name: str) -> bool:
    """Whether this log is on the exclusion list."""
    return bool(log_name) and log_name in excluded_logs()
