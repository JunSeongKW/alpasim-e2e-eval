"""자차의 실제 자세(pitch·roll)를 USDZ 에서 읽어, 수평 좌표계와 rig 좌표계를 잇는다.

NuRec 로그의 자차 pose 는 2D 다 -- `(x, y, yaw)` 뿐이고 pitch 와 roll 이 없다.
그래서 `gt_boxes` 의 z 는 **중력 기준 수평 좌표계** 값이고, 내리막에서는 앞쪽
박스가 아래로 내려간다.  반면 카메라 외부 파라미터(`sensor2lidar_*`)는 USDZ
캘리브레이션에서 온 **rig 좌표계** 값이고, rig 는 차와 함께 기울어져 있다.

둘을 그대로 합쳐 투영하면 오차가 **거리 x 경사**로 커진다.  clip `bbc85cf8`
(경사 -5%, pitch -3.35 deg)에서 138 m 앞 박스가 7.7 m 아래에 찍혔고, 카메라에서는
반대 차도 차량 아래 도로면에 박스가 그려졌다.  경사 2% 넘는 clip 이 21.9%,
5% 넘는 clip 이 4.6% 다 (50 m 앞 각각 1.0 m, 2.5 m 오차).

같은 `lidar2img` 를 BEVFormer 프론트엔드가 쓴다 -- BEV 격자점을 이미지에 투영해
피처를 뽑으므로, 이 어긋남은 시각화만이 아니라 학습에도 들어간다.

`T_rig_worlds` 가 6-DoF 자차 pose 이므로, 수평 좌표계 점을 rig 좌표계로 옮기는
행렬은

    C = inv(T_rig_worlds[t]) @ ego2global[t]

이고, 투영은 `lidar2img @ C` 가 된다.  검증: 이 보정을 넣으면 박스가 카메라의
차량에 정확히 붙는다 (`scripts/nurec/video_annotation_check.py --pitch-correct`).
"""
from __future__ import annotations

import json
import os
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

DEFAULT_USDZ_ROOT = "/home/irteam/data/nurec/sample_set/26.04_release"
MAX_TIMESTAMP_GAP_US = 300_000
"""이보다 멀리 떨어진 pose 는 쓰지 않는다.  rig pose 는 약 0.1 s 간격이라
0.3 s 를 넘어가면 그 프레임의 자세가 아니다."""


def usdz_root() -> Path:
    return Path(os.environ.get("NUREC_USDZ_ROOT", DEFAULT_USDZ_ROOT))


def clip_uuid(log_name: str) -> str:
    """`nurec-<uuid>` 또는 `<uuid>` 를 uuid 로."""
    return log_name[6:] if log_name.startswith("nurec-") else log_name


@lru_cache(maxsize=64)
def load_rig_poses(log_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(timestamps_us, T[N, 4, 4]) -- rig -> world.  없으면 None."""
    uuid = clip_uuid(log_name)
    path = usdz_root() / uuid / f"{uuid}.usdz"
    if not path.is_file():
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            rig = json.loads(archive.read("rig_trajectories.json"))
    except (KeyError, zipfile.BadZipFile, json.JSONDecodeError):
        return None
    entries = rig.get("rig_trajectories") or []
    if not entries:
        return None
    entry = entries[0]
    poses = np.asarray(entry.get("T_rig_worlds") or [], dtype=np.float64)
    stamps = np.asarray(entry.get("T_rig_world_timestamps_us") or [], dtype=np.int64)
    if poses.ndim != 3 or poses.shape[0] != stamps.shape[0] or stamps.size == 0:
        return None
    return stamps, poses


def horizontal_to_rig(log_name: str, timestamp_us: int, ego2global) -> Optional[np.ndarray]:
    """수평 ego 좌표계 -> rig 좌표계 4x4.  USDZ 가 없으면 None (보정 안 함)."""
    loaded = load_rig_poses(log_name)
    if loaded is None:
        return None
    stamps, poses = loaded
    index = int(np.argmin(np.abs(stamps - int(timestamp_us))))
    if abs(int(stamps[index]) - int(timestamp_us)) > MAX_TIMESTAMP_GAP_US:
        return None
    return np.linalg.inv(poses[index]) @ np.asarray(ego2global, dtype=np.float64)


TILT_TABLE = Path(__file__).resolve().parents[3] / "assets/nurec/rig_tilt.npz"


@lru_cache(maxsize=1)
def _tilt_table():
    """(log_name, timestamp) -> 보정행렬.  학습 중 USDZ 를 열지 않기 위한 표.

    clip 마다 30 MB zip 을 여는 것은 dataloader 가 감당할 수 없어서,
    `scripts/nurec/build_rig_tilt.py` 가 미리 뽑아 둔 것을 읽는다.
    """
    if not TILT_TABLE.is_file():
        return None
    data = np.load(TILT_TABLE, allow_pickle=False)
    keys = {}
    names = data["log_names"]
    stamps = data["timestamps"]
    for index in range(len(stamps)):
        keys[(str(names[index]), int(stamps[index]))] = index
    return keys, data["corrections"]


def tilt_for(log_name: str, timestamp_us: int) -> Optional[np.ndarray]:
    """미리 뽑아 둔 보정행렬.  표가 없거나 그 프레임이 없으면 None."""
    table = _tilt_table()
    if table is None:
        return None
    keys, mats = table
    index = keys.get((str(log_name), int(timestamp_us)))
    return None if index is None else mats[index].astype(np.float64)


def pitch_roll_deg(correction: np.ndarray) -> Tuple[float, float]:
    """보정행렬에서 pitch·roll 을 꺼낸다 (진단용)."""
    r = np.asarray(correction, dtype=np.float64)[:3, :3]
    pitch = float(np.degrees(np.arcsin(-np.clip(r[2, 0], -1.0, 1.0))))
    roll = float(np.degrees(np.arctan2(r[2, 1], r[2, 2])))
    return pitch, roll
