# Setup

```bash
source setup_nurec.sh
```

It prints one line per input and names anything missing. `pdm aggregate` reading
`not yet` is normal until `generate_epdms.sh` finishes — training reads the
per-token directory, not the aggregate pickle.

## What it sets, and why each one exists

| variable | value | note |
|---|---|---|
| `NAVSIM_DEVKIT_ROOT` | this tree | previously `../DriveSuprim`, a different copy of the same package |
| `NAVSIM_EXP_ROOT` | `./exp` → `../exp_v2` | checkpoints and hydra run snapshots |
| `NUREC_PREPARED_ROOT` | `./data/prepared` | K377 logs + `index.json` |
| `NUREC_SOURCE_ROOT` | `./data/prepared_source` | un-rectified root; maps/routes/metric_cache live here |
| `NUREC_SENSOR_ROOT` | `./data/images` | 512x256 rectified PNGs |
| `NUREC_MAP_ROOT` | `./data/prepared_source/maps` | `nurec:<clip>` map bundles |
| `NUREC_CALIBRATION` | `./assets/nurec/calibration.json` | per-clip `T_sensor_rig` + f-theta model |
| `NUREC_VOCAB_PATH` | `./assets/nurec/vocab/test_4096_kmeans.npy` | 4096 trajectory anchors |
| `NUREC_ORI_PDM_SCORE_DIR` | `./data/epdms/.../per_token` | what the planning loss actually reads |
| `NCCL_SOCKET_IFNAME` | auto | see below |

## The NCCL interface

This login environment ships `NCCL_SOCKET_IFNAME=eno1`, and `eno1` is DOWN with
no address. NCCL then fails with `Bootstrap : no socket interface found` before
the first step, single rank or not. `setup_nurec.sh` keeps an inherited value
only if the interface is up and addressed, and otherwise takes the interface the
default route uses. It says so when it overrides.

## Symlinks rather than copies

`./data/*` are symlinks. Three reasons, in order of weight:

1. `data/epdms` is being written right now by a long scoring run. Moving it
   would break that run.
2. `data/images` is 17 GB and `exp` is 83 GB.
3. `data/renders_raw` and `data/usdz` are not ours — the render pipeline owns
   them and is still writing to `renders_raw`.

Only `assets/nurec` is copied, because it is 7 MB and this tree owns it.

To point at a different dataset, either re-point the symlink or export the
variable before sourcing — every one of them honours an existing value.
