"""Per-clip ego vehicle box for NuRec clips, read from a side-car table.

AlpaSim scores collision and offroad against the *recording rig's* own box --
``rig_trajectories.json``'s ``rig_bbox`` inside each ``<clip>.usdz`` -- not
against a fixed vehicle.  Across the 1,607 clips of the 26.04 release there are
four distinct rigs, so one hard-coded size is wrong for 41% of them:

    5.207 x 2.157 x 1.823   centroid x 1.4675    949 clips  (59.1%)
    4.688 x 1.999 x 1.439   centroid x 1.3120    339 clips  (21.1%)
    5.393 x 2.109 x 1.503   centroid x 1.3965    244 clips  (15.2%)
    5.255 x 2.130 x 1.494   centroid x 1.4345     75 clips  ( 4.7%)

All four are narrower than nuPlan's Pacifica (2.297 m), which is what navsim
falls back to.  Scoring a 2.0 m wide rig as a 2.3 m wide Pacifica turns a third
of the human trajectory's genuine near-misses into at-fault collisions -- see
``docs/nurec/NC_HUMAN_FILTER.md`` section 6.

``NuRecMap`` exposes ``rig_bbox`` when the map bundle was extracted by a version
of ``nurec_map_data.py`` that writes it.  Bundles produced before that carry no
such key, and re-extracting them costs a pass over 2.7 TB of USDZ, so this table
-- built once by ``scripts/nurec/analysis/`` from the same source -- stands in.
The map bundle always wins when it has the value.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

TABLE_ENV = "NUREC_RIG_BBOX_TABLE"
DEFAULT_TABLE = "assets/nurec/rig_bbox.json"


def _table_path() -> Optional[Path]:
    override = os.environ.get(TABLE_ENV)
    if override:
        return Path(override)
    root = os.environ.get("AXE_ROOT") or os.environ.get("NAVSIM_DEVKIT_ROOT")
    if not root:
        return None
    return Path(root) / DEFAULT_TABLE


@lru_cache(maxsize=1)
def _table() -> Mapping[str, Mapping[str, Any]]:
    path = _table_path()
    if path is None or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text()).get("clips", {})
    except (OSError, ValueError):
        return {}


def rig_bbox_for_log(log_name: str) -> Optional[Mapping[str, Any]]:
    """``rig_bbox`` for a NuRec log, or None when the clip is not in the table.

    Log names carry a ``nurec-`` prefix that clip directories do not.
    """
    if not log_name:
        return None
    return _table().get(log_name.removeprefix("nurec-")) or _table().get(log_name)
