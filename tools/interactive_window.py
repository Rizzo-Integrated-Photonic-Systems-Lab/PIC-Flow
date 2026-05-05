#!/usr/bin/env python3
"""
Desktop window for live PIC-Flow parametric inference (geometry + |E_z|).

Uses Tkinter + Matplotlib (stdlib + existing deps). Inference and Meep rasterization
run on a background thread so sliders stay responsive; updates are debounced so
dragging a slider batches work until you pause briefly.

Run from anywhere:

  python tools/interactive_window.py
  python tools/interactive_window.py --ckpt checkpoints/phase_residual_300.pt

Requires: same stack as notebooks/04_interactive.ipynb (torch, meep, matplotlib).
On WSL2, use a graphical session (WSLg) or X11 forwarding so Tk can open a window.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

for _p in (str(REPO_ROOT), str(REPO_ROOT / "Model"), str(REPO_ROOT / "tools"), str(REPO_ROOT / "FDTD")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_param_ranges(repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = repo / "FDTD" / "param_ranges.py"
    spec = importlib.util.spec_from_file_location("_pic_flow_param_ranges", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pr = getattr(mod, "PARAM_RANGES")
    ipt = getattr(mod, "INPUT_PORTS", None)
    if ipt is None:
        ipt = {
            "straight": [1, 2],
            "taper": [1, 2],
            "mmi": [1, 2],
            "sbend": [1, 2],
            "ybranch": [1, 2, 3],
            "directional_coupler": [1, 2],
            "euler_bend": [1, 2],
            "circular_bend": [1, 2],
            "crossing": [1, 2, 3, 4],
        }
    return pr, ipt


PARAM_RANGES, INPUT_PORTS = _load_param_ranges(REPO_ROOT)

try:
    import meep as mp

    mp.verbosity(0)
except ImportError:
    mp = None

from flow_matching import sample as fm_sample  # noqa: E402
from huggingface_hub import hf_hub_download, try_to_load_from_cache  # noqa: E402
from predict_parametric_device import (  # noqa: E402
    DEFAULT_CROP_X_PX,
    DEFAULT_CROP_Y_PX,
    DEFAULT_DPML,
    DEFAULT_RESOLUTION,
    DEFAULT_WAVELENGTH_UM,
    _build_cond_maps,
    _build_cond_vector,
    _checkpoint_state_dict,
    _build_model_from_checkpoint,
    _make_device_and_arrays,
)

import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

DEVICES = ("mmi", "ybranch", "directional_coupler")
DEVICE_LABELS = {"mmi": "2x2 MMI", "ybranch": "Y-branch", "directional_coupler": "Directional coupler"}
PARAM_QUANTUM = 0.025
_GEOM_CACHE: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray, tuple[float, float]]] = {}

_HF_REPO = "RizzoLab/PIC-Flow"
_HF_FILE = "checkpoints/phase_residual_300.pt"


def _resolve_ckpt(repo: Path, hf_file: str) -> Path:
    local = repo / Path(hf_file)
    if local.is_file():
        return local.resolve()
    cached = try_to_load_from_cache(repo_id=_HF_REPO, filename=hf_file)
    if cached is not None and cached is not False:
        return Path(str(cached)).resolve()
    return Path(
        str(hf_hub_download(repo_id=_HF_REPO, filename=hf_file, repo_type="model"))
    ).resolve()


def rasterize(
    device_type: str,
    params_tuple: tuple[tuple[str, float], ...],
    source_port: int,
    wavelength_um: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    key = (device_type, params_tuple, int(source_port), float(wavelength_um))
    if key in _GEOM_CACHE:
        return _GEOM_CACHE[key]
    _, eps, src, _, _, cell = _make_device_and_arrays(
        device_type=device_type,
        params=dict(params_tuple),
        source_port=int(source_port),
        wavelength_um=float(wavelength_um),
        resolution=DEFAULT_RESOLUTION,
        dpml=DEFAULT_DPML,
        crop_x_px=DEFAULT_CROP_X_PX,
        crop_y_px=DEFAULT_CROP_Y_PX,
    )
    eps = np.asarray(eps, dtype=np.float32)
    src = np.asarray(src, dtype=np.float32)
    cell = (float(cell[0]), float(cell[1]))
    _GEOM_CACHE[key] = (eps, src, cell)
    return _GEOM_CACHE[key]


def infer_field(
    eps: np.ndarray,
    src_mask: np.ndarray,
    cell_size: tuple[float, float],
    num_steps: int,
    seed: int,
    wavelength_um: float,
    *,
    model: torch.nn.Module,
    stats: dict[str, Any],
    ckpt_args: Any,
    time_grid: str,
    torch_device: torch.device,
) -> np.ndarray:
    dx_um = cell_size[0] / eps.shape[1]
    dy_um = cell_size[1] / eps.shape[0]
    cond_maps = _build_cond_maps(
        eps,
        src_mask,
        stats=stats,
        ckpt_args=ckpt_args,
        device=torch_device,
        dx_um=dx_um,
        dy_um=dy_um,
    )
    cond_v = _build_cond_vector(wavelength_um, stats, device=torch_device)
    lam = torch.tensor([[wavelength_um]], device=torch_device, dtype=torch.float32)

    gen = torch.Generator(device=torch_device)
    gen.manual_seed(int(seed))
    x0 = torch.randn((1, 2, eps.shape[0], eps.shape[1]), device=torch_device, dtype=torch.float32, generator=gen)
    use_amp = torch_device.type == "cuda"
    with torch.no_grad():
        with torch.autocast(device_type=torch_device.type, dtype=torch.float16, enabled=use_amp):
            x1 = fm_sample(
                model,
                x0,
                num_steps=int(num_steps),
                cond_maps=cond_maps,
                cond=cond_v,
                lambda_um=lam,
                time_grid=time_grid,
                progress=False,
            )
    fn = x1.float()[0].cpu().numpy()
    ezr = fn[0] * float(stats["ez_real_std"]) + float(stats["ez_real_mean"])
    ezi = fn[1] * float(stats["ez_imag_std"]) + float(stats["ez_imag_mean"])
    return np.abs(ezr + 1j * ezi).astype(np.float32)


class PicFlowInteractiveApp:
    def __init__(
        self,
        *,
        ckpt_path: Path,
        debounce_ms: int,
        wavelength_um: float,
    ) -> None:
        self._debounce_ms = int(debounce_ms)
        self._wavelength_um = float(wavelength_um)

        self._torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(str(ckpt_path), map_location=self._torch_device, weights_only=False)
        self._stats = ckpt["stats"]
        self._ckpt_args = ckpt.get("args")
        self._time_grid = (
            str(getattr(self._ckpt_args, "time_grid", "linear")) if self._ckpt_args is not None else "linear"
        )

        self.model = _build_model_from_checkpoint(ckpt, device=self._torch_device)
        _, state = _checkpoint_state_dict(ckpt, use_ema=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        self._job_serial = 0
        self._debounce_after_id: str | None = None
        self._infer_lock = threading.Lock()

        self.root = tk.Tk()
        self.root.title("PIC-Flow — interactive inference")
        self.root.minsize(960, 620)

        self._device_labels = [DEVICE_LABELS[d] for d in DEVICES]

        ctrl = ttk.Frame(self.root, padding=8)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(ctrl, text="Device").pack(side=tk.LEFT, padx=(0, 4))
        self.cmb_device = ttk.Combobox(
            ctrl,
            values=self._device_labels,
            width=22,
            state="readonly",
        )
        self.cmb_device.set(self._device_labels[2])
        self.cmb_device.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(ctrl, text="Source port").pack(side=tk.LEFT, padx=(0, 4))
        self.var_port = tk.StringVar(value="1")
        self.cmb_port = ttk.Combobox(ctrl, textvariable=self.var_port, width=6, state="readonly")
        self.cmb_port.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(ctrl, text="Euler steps").pack(side=tk.LEFT, padx=(0, 4))
        self.var_steps = tk.IntVar(value=20)
        sp_steps = ttk.Spinbox(ctrl, from_=1, to=200, textvariable=self.var_steps, width=8)
        sp_steps.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(ctrl, text="Seed").pack(side=tk.LEFT, padx=(0, 4))
        self.var_seed = tk.IntVar(value=0)
        sp_seed = ttk.Spinbox(ctrl, from_=0, to=999999, textvariable=self.var_seed, width=8)
        sp_seed.pack(side=tk.LEFT, padx=(0, 12))

        self.var_live = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Smooth live update", variable=self.var_live).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(ctrl, text="Refresh now", command=lambda: self._kick_job(immediate=True)).pack(side=tk.RIGHT)

        self.slider_frame = ttk.LabelFrame(self.root, text="Geometry parameters (µm)", padding=8)
        self.slider_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))

        self.param_vars: dict[str, tk.DoubleVar] = {}

        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.fig = Figure(figsize=(11.0, 3.6), dpi=100)
        self.ax_eps = self.fig.add_subplot(1, 2, 1)
        self.ax_mag = self.fig.add_subplot(1, 2, 2)
        self.fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.12, wspace=0.25)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.status = ttk.Label(self.root, text="Ready — adjust sliders (updates while dragging when enabled).", anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)

        self.cmb_device.bind("<<ComboboxSelected>>", self._on_device_label_change)
        self.cmb_port.bind("<<ComboboxSelected>>", lambda _e: self._schedule_update())
        self.var_steps.trace_add("write", lambda *a: self._schedule_update())
        self.var_seed.trace_add("write", lambda *a: self._schedule_update())

        self._on_device_label_change()

    def _current_device_key(self) -> str:
        label = self.cmb_device.get()
        for d in DEVICES:
            if DEVICE_LABELS[d] == label:
                return d
        return "directional_coupler"

    def _on_device_label_change(self, _evt=None) -> None:
        dev = self._current_device_key()
        ports = INPUT_PORTS.get(dev, [1])
        self.cmb_port["values"] = [str(p) for p in ports]
        self.var_port.set(str(ports[0]))
        self._rebuild_param_sliders(dev)


    def _rebuild_param_sliders(self, device_type: str) -> None:
        for w in self.slider_frame.winfo_children():
            w.destroy()
        self.param_vars.clear()

        ranges = PARAM_RANGES[device_type]
        col = ttk.Frame(self.slider_frame)
        col.pack(fill=tk.X)
        half = max(1, (len(ranges) + 1) // 2)

        names = list(ranges.keys())
        for i, name in enumerate(names):
            lo, hi = ranges[name]
            mid = round(((lo + hi) * 0.5) / PARAM_QUANTUM) * PARAM_QUANTUM
            v = tk.DoubleVar(value=mid)
            self.param_vars[name] = v

            grid_col = i // half
            row = i % half
            sub = ttk.Frame(col)
            sub.grid(row=row, column=grid_col, sticky=tk.EW, padx=(0 if grid_col == 0 else 16, 8), pady=2)

            ttk.Label(sub, text=f"{name}", width=18).pack(side=tk.LEFT)

            tk_scale = tk.Scale(
                sub,
                from_=lo,
                to=hi,
                resolution=PARAM_QUANTUM,
                orient=tk.HORIZONTAL,
                variable=v,
                length=340,
                showvalue=True,
                command=lambda _x, vn=name: self._on_slider_moved(vn),
            )
            tk_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ncols = max(1, math.ceil(len(names) / half))
        for c in range(ncols):
            col.columnconfigure(c, weight=1)

    def _on_slider_moved(self, _name: str) -> None:
        self._schedule_update()

    def _schedule_update(self) -> None:
        if not self.var_live.get():
            return
        if self._debounce_after_id is not None:
            self.root.after_cancel(self._debounce_after_id)
        self._debounce_after_id = self.root.after(self._debounce_ms, self._debounce_fire)

    def _debounce_fire(self) -> None:
        self._debounce_after_id = None
        self._kick_job(immediate=False)

    def _kick_job(self, *, immediate: bool) -> None:
        if immediate and self._debounce_after_id is not None:
            self.root.after_cancel(self._debounce_after_id)
            self._debounce_after_id = None

        self._job_serial += 1
        serial = self._job_serial

        self.status.config(text="Working… (geometry + inference)")

        def work(sn: int) -> None:
            with self._infer_lock:
                if sn != self._job_serial:
                    return
                try:
                    dev = self._current_device_key()
                    port = int(self.var_port.get())
                    steps = int(self.var_steps.get())
                    seed = int(self.var_seed.get())
                    params_t = tuple(sorted((n, float(v.get())) for n, v in self.param_vars.items()))
                except (tk.TclError, ValueError):
                    return
                try:
                    t0 = time.perf_counter()
                    eps, src, cell = rasterize(dev, params_t, port, self._wavelength_um)
                    t_geom = time.perf_counter() - t0
                    if sn != self._job_serial:
                        return
                    t1 = time.perf_counter()
                    mag = infer_field(
                        eps,
                        src,
                        cell,
                        steps,
                        seed,
                        self._wavelength_um,
                        model=self.model,
                        stats=self._stats,
                        ckpt_args=self._ckpt_args,
                        time_grid=self._time_grid,
                        torch_device=self._torch_device,
                    )
                    t_inf = time.perf_counter() - t1
                    cached = t_geom < 0.05
                    self.root.after(
                        0,
                        lambda ss=sn, e=eps, s=src, m=mag, st=steps, tg=t_geom, ti=t_inf, ca=cached: self._apply_result(
                            ss, e, s, m, st, tg, ti, ca
                        ),
                    )
                except Exception as e:  # pragma: no cover
                    err = str(e)
                    self.root.after(
                        0,
                        lambda ssn=sn, er=err: self._apply_error(ssn, er),
                    )

        threading.Thread(target=work, args=(serial,), daemon=True).start()

    def _apply_result(
        self,
        serial: int,
        eps: np.ndarray,
        src: np.ndarray,
        mag: np.ndarray,
        steps: int,
        t_geom: float,
        t_inf: float,
        cached: bool,
    ) -> None:
        if serial != self._job_serial:
            return

        self.ax_eps.clear()
        self.ax_eps.imshow(eps, origin="lower", cmap="viridis", interpolation="nearest", aspect="equal")
        green = np.ma.masked_where(src <= 0.5, np.ones_like(src, dtype=np.float32))
        self.ax_eps.imshow(green, origin="lower", cmap="Greens", vmin=0, vmax=1, alpha=0.55, interpolation="nearest", aspect="equal")
        tag = " (cached)" if cached else ""
        self.ax_eps.set_title(f"Geometry ε_r  (raster {t_geom * 1000:.0f} ms{tag})")
        self.ax_eps.set_xticks([])
        self.ax_eps.set_yticks([])

        self.ax_mag.clear()
        vmax = float(np.percentile(mag, 99.5))
        vmax = max(vmax, 1e-6)
        self.ax_mag.imshow(mag, origin="lower", cmap="magma", vmin=0, vmax=vmax, interpolation="nearest", aspect="equal")
        self.ax_mag.set_title(f"PIC-Flow |E_z|  ({steps} steps, {t_inf * 1000:.0f} ms)")
        self.ax_mag.set_xticks([])
        self.ax_mag.set_yticks([])

        self.canvas.draw_idle()
        self.status.config(
            text=f"Done — geometry {t_geom * 1000:.0f} ms, inference {t_inf * 1000:.0f} ms (live debounce {self._debounce_ms} ms)"
        )

    def _apply_error(self, serial: int, err: str) -> None:
        if serial != self._job_serial:
            return
        self.status.config(text=f"Error: {err}")

    def run(self) -> None:
        if mp is None:
            self.status.config(text="Error: meep not installed — conda install -c conda-forge pymeep")
        self.root.mainloop()


def main() -> None:
    ap = argparse.ArgumentParser(description="PIC-Flow interactive desktop UI")
    ap.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Checkpoint .pt (default: repo checkpoints/phase_residual_300.pt or HF cache)",
    )
    ap.add_argument("--debounce-ms", type=int, default=180, help="Delay after slider move before running (ms)")
    ap.add_argument("--wavelength-um", type=float, default=DEFAULT_WAVELENGTH_UM)
    args = ap.parse_args()

    if args.ckpt:
        ckpt_path = Path(args.ckpt).expanduser().resolve()
        if not ckpt_path.is_file():
            print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
            sys.exit(1)
    else:
        ckpt_path = _resolve_ckpt(REPO_ROOT, _HF_FILE)

    print(f"Checkpoint: {ckpt_path}")
    app = PicFlowInteractiveApp(
        ckpt_path=ckpt_path,
        debounce_ms=args.debounce_ms,
        wavelength_um=args.wavelength_um,
    )
    app.run()


if __name__ == "__main__":
    main()
