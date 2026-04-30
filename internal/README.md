# Internal — research-side material

Everything here is preserved for reproducibility of the paper's specific runs and for the
authors' own development workflow. **Public users do not need anything in this folder.**

## Layout

| Path | What |
|---|---|
| `slurm/` | SLURM job configs grouped by purpose: `train/`, `finetune/`, `sweep/`, `data/`, `neff/`. |
| `runs/` | Run artifacts: `wandb/`, `logs/`, `outputs/`, prior `checkpoints_resume/`, generated figure dumps. **Gitignored.** |
| `notebooks/` | Older exploration notebooks (e.g. `FDTD_main.ipynb`); not maintained as user-facing material. |
| `debug/` | One-off debug and probe scripts: residual-floor checks, modal S-param tests, import smoke tests, the `tidy3d` job killer. |
| `data_tools/` | Dataset-side helpers: preview scripts, geometry validators, single-device synthesis tools, GDS fixtures. |
| `archived/` | Snapshot-only legacy code: pre-unified per-device sweepers (`*_old.py`, `mmi_2x2*.py`, `coupler_sweep.py`, etc.). Kept for reference; not imported by any public path. |
| `train_subset/` | Quick-iteration subset trainer used during development. |
| `claude/` | `CLAUDE.md` files from the repo root, `Model/`, and `FDTD/` (per-directory development notes). |

## SLURM scripts

The three paper runs live at:

- `slurm/train/train_real_unet_fm_only.slurm`
- `slurm/train/train_real_unet_phase.slurm`
- `slurm/train/train_real_unet_phase_residual.slurm` *(or `_fm_residual.slurm`)*

These were originally written against the full pre-cleanup CLI; the deprecated flags
(`--lambda-endpoint`, `--lambda-phase-grad`, `--joint-training`, `--unroll-steps`,
`--phase-amp-tau`, etc.) were stripped after the public-facing args trim. Hardcoded paths
(`/dartfs-hpc/...`, `/dartfs/rc/lab/R/RizzoA/...`) and the `v100_preemptable` partition
will need to be edited for any other site.

## What's intentionally not here

- `Data/` — gitignored at the repo root, not relocated.
- The current public training script `Model/train.py`, the public sampler `Model/sample.py`,
  and everything under `tools/` — those are the user-facing surface and stay at the top level.
