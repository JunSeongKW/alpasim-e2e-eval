# SPDX-License-Identifier: Apache-2.0
"""Which threshold makes AlpaSim's route->command rule speak NAVSIM's language?

    python calibrate_command.py --logs .../navsim_logs/trainval [--n-logs 400]

DrivoR was trained on NAVSIM's driving_command (left / straight / right /
unknown; "based solely on the desired route ... cover turns, lane changes and
sharp curves", docs/agents.md). In the e2e challenge the driver gets route
waypoints that start route_start_offset_m = 40 m ahead (dev preset), and the
official rule (alpasim_driver.navigation) reads the lateral offset of the first
waypoint at least min_lookahead ahead against a threshold. Training data only:
for every NAVSIM trainval frame, the recorded future path stands in for the
route, the official rule is applied at the 40 m point, and agreement with the
logged driving_command is tabulated per (lookahead, threshold). Lookaheads
below 40 m act like 40 m in the challenge (the first waypoint is ~40 m out), so
the grid starts at 40. Balanced accuracy (mean recall over L/S/R) is the
criterion; plain agreement is dominated by the 70% straight frames and rises
with any threshold. No AlpaSim scene is used.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

LOOKAHEADS = [10.0, 20.0, 40.0, 50.0, 60.0, 80.0]
THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--n-logs", type=int, default=400)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    logs = sorted(a.logs.glob("*.pkl"))
    logs = logs[:: max(1, len(logs) // a.n_logs)][: a.n_logs]
    dys, cmds = {L: [] for L in LOOKAHEADS}, {L: [] for L in LOOKAHEADS}
    for path in logs:
        frames = pickle.load(path.open("rb"))
        pos = np.array([f["ego2global_translation"][:2] for f in frames], dtype=np.float64)
        yaw = np.array([Quaternion(*f["ego2global_rotation"]).yaw_pitch_roll[0] for f in frames])
        for i, f in enumerate(frames):
            cmd = int(np.argmax(f["driving_command"]))
            c, s = np.cos(yaw[i]), np.sin(yaw[i])
            rel = (pos[i + 1:] - pos[i]) @ np.array([[c, -s], [s, c]])   # into ego frame
            if len(rel) < 2:
                continue
            dist = np.linalg.norm(rel, axis=1)
            for L in LOOKAHEADS:
                k = int(np.argmax(dist >= L))
                if dist[k] < L:
                    continue   # recorded future shorter than L (slow / stopped)
                dys[L].append(rel[k, 1])
                cmds[L].append(cmd)
    names = ["left", "straight", "right", "unknown"]
    rows = []
    for L in LOOKAHEADS:
        dy, cm = np.array(dys[L]), np.array(cmds[L])
        known = cm < 3
        print(f"lookahead {L:.0f} m: {len(cm)} frames; NAVSIM shares",
              {names[c]: round(float((cm == c).mean()), 3) for c in range(4)})
        for th in THRESHOLDS:
            pred = np.where(dy > th, 0, np.where(dy < -th, 2, 1))
            acc = float((pred[known] == cm[known]).mean())
            recall = [float((pred[cm == t] == t).mean()) for t in range(3)]
            bal = float(np.mean(recall))
            rows.append({"lookahead_m": L, "threshold_m": th, "agreement": round(acc, 4),
                         "balanced_accuracy": round(bal, 4), "recall_lsr": [round(r, 4) for r in recall]})
            print(f"  th {th:.1f}: agree {acc:.3f}  balanced {bal:.3f}  recall L/S/R {[round(r, 3) for r in recall]}")
    best = max((r for r in rows if r["lookahead_m"] >= 40), key=lambda r: r["balanced_accuracy"])
    print("best feasible (lookahead >= 40):", best)
    if a.out:
        a.out.write_text(json.dumps({"rows": rows, "best_feasible": best}, indent=1) + "\n")


if __name__ == "__main__":
    main()
