#!/usr/bin/env python3
"""Validate calibration-fixed driver diagnostics after inference has started."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--min-coverage", type=float, default=95.0)
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")
    failures: list[str] = []

    required = {
        "exact checkpoint load": r"exact checkpoint load:\s*missing=0\s+unexpected=0",
        "K377 rectifier": r"K377 rectifier ready",
        "BEV tensor contract": (
            r"BEV INPUT:.*imgs=\(1, 3, 3, 3, 256, 512\).*"
            r"lidar2img=\(1, 3, 3, 4, 4\)"
        ),
        "positive route": r"route_valid=([1-9][0-9]*)",
    }
    for label, pattern in required.items():
        if not re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            failures.append(f"missing {label}")

    for camera in ("CAM_L0", "CAM_F0", "CAM_R0"):
        pattern = (
            rf"virtual pinhole:\s*{camera}\s+"
            rf"K=\(fx=377\.000,\s*fy=377\.000,\s*"
            rf"cx=256\.000,\s*cy=128\.000\)"
        )
        if not re.search(pattern, text):
            failures.append(f"missing exact K377 log for {camera}")

    coverages = [
        (float(any_camera), float(mean))
        for any_camera, mean in re.findall(
            r"BEV cells reached by any camera:\s*([0-9.]+)%\s*"
            r"\(cell-camera-anchor mean\s*([0-9.]+)%\)",
            text,
        )
    ]
    if not coverages:
        failures.append("missing BEV visibility report; wait for first inference")
    else:
        for index, (any_camera, mean) in enumerate(coverages, start=1):
            if any_camera < args.min_coverage:
                failures.append(
                    f"driver {index} BEV coverage {any_camera:.3f}% "
                    f"< {args.min_coverage:.1f}%"
                )
            if mean < 5.0:
                failures.append(
                    f"driver {index} cell-camera-anchor mean {mean:.3f}% is near zero"
                )

    forbidden = {
        "calibration mismatch": r"CalibrationMismatch|does not belong to this image",
        "unsupported H100 kernel": r"no kernel image is available",
        "CUDA OOM": r"CUDA out of memory",
        "checkpoint mismatch": r"missing=[1-9][0-9]*|unexpected=[1-9][0-9]*",
        "Python traceback": r"Traceback \(most recent call last\)",
    }
    for label, pattern in forbidden.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            failures.append(f"found {label}")

    print(f"log: {args.log}")
    if coverages:
        for index, (any_camera, mean) in enumerate(coverages, start=1):
            print(
                f"BEV visibility #{index}: any-camera={any_camera:.3f}% "
                f"cell-camera-anchor={mean:.3f}%"
            )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("OK: checkpoint, K377, BEV visibility, tensor shape, and route are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
