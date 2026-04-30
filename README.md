# Rayfield — PHASE: Physics-Aware Surrogate for Electromagnetics

A neural surrogate that replaces FDTD simulation for 2D photonic device field prediction.
PHASE pairs a real-valued U-Net with **conditional flow matching** and a **Helmholtz residual
loss** to generate complex electromagnetic fields ($E_z$) for parameterized silicon-photonic
devices given their permittivity map, source-port mask, and wavelength. See [`paper.tex`](paper.tex)
for the full method and benchmarks.

<p align="center">
  <img src="assets/denoising_trajectory.gif"
       alt="PHASE denoising trajectory: 400-step Euler integration from Gaussian noise to the predicted E_z field, with the FDTD ground truth shown alongside for reference."
       width="760">
</p>

The animation shows what the model actually does: starting from Gaussian noise (left, $t=0$),
PHASE integrates a learned velocity field along $t \in [0, 1]$ until the field converges to a
prediction (left, $t=1$) that matches the FDTD reference (right). The whole 400-step integration
runs in under a second on a single A100; with 20 steps it takes ≈ 444 ms and still hits 3 %
Helmholtz compliance, ~10× faster than 16-thread CPU FDTD on the same node.

> Regenerate the GIF: `python tools/denoising_trajectory.py` then copy
> `outputs/denoising/trajectory.gif` to `assets/denoising_trajectory.gif`.

```
input:  (epsilon, source mask, wavelength)  -->  PHASE U-Net (Euler/Heun ODE sampler)  -->  E_z(x,y)
```

Trained dataset covers three device families at 1.55 µm (multimode interferometers, Y-branches,
and directional couplers — 22 500 FDTD simulations total).

---

## Install

```bash
git clone <repo-url> rayfield
cd rayfield

# Core deps. PyTorch must match your CUDA — install separately if pip's default doesn't fit:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Optional: only needed if you generate FDTD data yourself.
# Meep is best installed via conda-forge: `conda install -c conda-forge pymeep`.
```

Tested with Python 3.10+, PyTorch 2.x, CUDA 11.8/12.1.

---

## Quickstart

The fastest path to seeing the model work:

```bash
# 1) install (above)
# 2) drop a checkpoint at ./checkpoints/phase_300.pt  (link in paper / contact authors)
# 3) open the inference notebook
jupyter lab notebooks/03_inference.ipynb
```

The four notebooks cover the full lifecycle:

| Notebook | What it shows |
|---|---|
| [`notebooks/01_dataset_generation.ipynb`](notebooks/01_dataset_generation.ipynb) | Generate a small FDTD dataset (~10 geometries/family) with `FDTD/unified_sweep.py`. |
| [`notebooks/02_training.ipynb`](notebooks/02_training.ipynb) | Smoke-train the U-Net on a tiny subset; loss curves. |
| [`notebooks/03_inference.ipynb`](notebooks/03_inference.ipynb) | Load the FM+phase+residual checkpoint, predict $E_z$ on a test sample, plot triptych + compliance. |
| [`notebooks/04_inverse_design.ipynb`](notebooks/04_inverse_design.ipynb) | Gradient-based inverse design: target a 50/50 Y-branch split through the trained surrogate. |

### Training (full)

```bash
# Single GPU
python Model/train.py --data-root Data/ --epochs 300 --batch-size 4

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 Model/train.py --data-root Data/ --batch-size 16

# Three paper runs (FM, FM+phase, FM+phase+residual)
python Model/train.py --data-root Data/ --lambda-residual 0 --lambda-phase 0
python Model/train.py --data-root Data/ --lambda-residual 0 --lambda-phase 0.1
python Model/train.py --data-root Data/ --lambda-residual 1.0 --lambda-phase 0.1
```

`python Model/train.py --help` lists all flags. The defaults reproduce the paper's three runs at
the appropriate `--lambda-*` weights.

### Inference (CLI)

```bash
# Single device folder (eps.npy, src_mask.npy, sparams.npz with wavelength_um)
python Model/sample.py --device-dir path/to/device/ --ckpt checkpoints/phase_300.pt

# Parametric device (no FDTD needed; specify geometry directly)
python tools/predict_parametric_device.py --device directional_coupler \
    --gap 0.175 --L_c 5.97 --wavelength 1.55 --ckpt checkpoints/phase_300.pt

# Full test-set sweep with metric histograms
python tools/test_set_histograms.py --ckpt checkpoints/phase_300.pt --num-steps 100
```

### Dataset generation (FDTD)

```bash
# Generate the unified 3-family dataset (~24 hours on a single CPU node with 16 threads)
python FDTD/unified_sweep.py --output-dir Data/ \
    --devices mmi,ybranch,directional_coupler \
    --num-samples 7500 --wavelengths 1.55
```

See [`Data/README.md`](Data/README.md) for the dataset layout.

---

## Repo layout

```
rayfield/
├── Model/         neural network + training loop + flow matching
├── FDTD/          dataset generation (Meep-based) + per-device geometries
├── tools/         inference, inverse design, benchmarks, evaluation
├── notebooks/     hands-on walkthroughs (start here)
├── examples/      sample evaluation indices
├── paper.tex      manuscript
├── references.bib
├── Data/          (gitignored) FDTD ground-truth shards
└── internal/      research-side material: SLURM configs, run logs, archived experiments
```

Public users don't need anything in `internal/`; it's preserved for reproducibility of the
paper's specific runs.

---

## Citation

If you use this code, please cite:

```bibtex
@article{Quaratiello2026PHASE,
  author  = {Joseph Quaratiello and Anthony Rizzo},
  title   = {A Physics-Embedded Flow-Matching Model for Electromagnetic Prediction
             of Silicon Photonic Devices},
  journal = {arXiv},
  year    = {2026}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is included.

---

## License

MIT — see [`LICENSE`](LICENSE).
