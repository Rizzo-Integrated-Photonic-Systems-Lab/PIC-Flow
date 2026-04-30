# Dataset

The `Data/` directory holds FDTD ground-truth shards consumed by training and evaluation.
It is **gitignored** (see [`.gitignore`](../.gitignore)) — generate or download it locally.

## Layout

```
Data/
└── unified_sweep_mmi_ybranch_dc_7500_each_1p55um/
    └── shards/
        ├── index.json
        ├── 0000.npz
        ├── 0001.npz
        └── ...
```

Each `.npz` shard packs many per-sample arrays (real/imag $E_z$, permittivity, source mask,
port masks, wavelength, S-parameters, geometry parameters). The `index.json` carries the
canonical train/val/test split that the paper's three runs share.

## Generate

```bash
python FDTD/unified_sweep.py \
    --output-dir Data/ \
    --devices mmi,ybranch,directional_coupler \
    --num-samples 7500 \
    --wavelengths 1.55
```

Wall time: ~24 h on a single 16-thread CPU node. See [`notebooks/01_dataset_generation.ipynb`](../notebooks/01_dataset_generation.ipynb)
for a smaller (~10 geometries/family) walkthrough.

Meep is required (install via `conda install -c conda-forge pymeep`).

## Download

Pre-generated shards may be released alongside the paper. Until then, run the generator above.
