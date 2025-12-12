import numpy as np

# Pick a sample folder
meta = np.load("Data/coupler_sweep/wgWidth0.380_gap0.109_wgLength5.1_bendLength6.0_leadGap1.78_lam1.59/grid_meta.npz")
print("grid_meta keys:", meta.files)
for k in meta.files:
    print(k, "=", meta[k])

sparams = np.load("Data/coupler_sweep/wgWidth0.380_gap0.109_wgLength5.1_bendLength6.0_leadGap1.78_lam1.59/sparams.npz")
print("sparams keys:", sparams.files)
for k in sparams.files:
    print(k, "shape:", sparams[k].shape)
    print(k, "value:", sparams[k])
