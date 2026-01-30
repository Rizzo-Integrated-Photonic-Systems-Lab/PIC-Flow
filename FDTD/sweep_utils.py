import json
import os
import uuid

import numpy as np


def latin_hypercube(n, d, seed=42):
    rng = np.random.default_rng(int(seed))
    cut = np.linspace(0.0, 1.0, int(n) + 1, dtype=np.float64)
    a = cut[:-1]
    b = cut[1:]
    u = rng.random((int(n), int(d))).astype(np.float64, copy=False)
    H = u * (b - a)[:, None] + a[:, None]
    for j in range(int(d)):
        rng.shuffle(H[:, j])
    return H.astype(np.float64, copy=False)


def quantize_01(x, x_min, x_max):
    xq = np.round(x * 100.0) / 100.0
    return np.clip(xq, x_min, x_max)


def parse_wavelengths(text):
    if text is None:
        return []
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [float(p) for p in parts]


def assign_splits(n, seed=123, ratios=(0.8, 0.1, 0.1)):
    n = int(n)
    if n <= 0:
        return []
    if len(ratios) != 3:
        raise ValueError("ratios must be a 3-tuple (train, val, test)")
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    n_test = n - n_train - n_val
    if n_test < 0:
        raise ValueError("invalid ratios")
    labels = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test, dtype=object)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(labels)
    return labels.tolist()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def make_geom_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex}"





