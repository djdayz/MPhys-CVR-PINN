
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_nifti(path):
    return nib.load(str(path))


def get_data(img):
    return np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)


def save_nifti_like(data, ref_img, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(data.astype(np.float32), ref_img.affine, ref_img.header)
    nib.save(out, str(out_path))


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device)
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        return torch.load(path, map_location=device, weights_only=False)


def build_long_etco2(extra_pre=31, extra_post=93):
    parts = [
        40.0 * np.ones(int(extra_pre), dtype=np.float32),
        40.0 * np.ones(120, dtype=np.float32),
        50.0 * np.ones(180, dtype=np.float32),
        40.0 * np.ones(120, dtype=np.float32),
        50.0 * np.ones(180, dtype=np.float32),
        40.0 * np.ones(121, dtype=np.float32),
        40.0 * np.ones(int(extra_post), dtype=np.float32),
    ]
    return np.concatenate(parts, axis=0)


def resample_signal_to_tr(signal, signal_dt, n_time, tr):
    t_src = np.arange(signal.size, dtype=np.float32) * float(signal_dt)
    t_dst = np.arange(n_time, dtype=np.float32) * float(tr)
    f = interp1d(
        t_src,
        signal,
        kind="linear",
        bounds_error=False,
        fill_value=(float(signal[0]), float(signal[-1])),
    )
    return np.asarray(f(t_dst), dtype=np.float32)


def _load_text_matrix(path):
    loaders = (
        lambda: np.genfromtxt(path, delimiter=",", dtype=np.float32),
        lambda: np.genfromtxt(path, delimiter=None, dtype=np.float32),
    )
    for loader in loaders:
        arr = loader()
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size == 0:
            continue
        if arr.ndim == 0 and not np.isfinite(arr):
            continue
        return arr
    raise ValueError(f"Could not parse EtCO2 file: {path}")


def load_etco2_from_file(path, sample_dt, time_column=0, value_column=1):
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes[-1:] == [".npy"]:
        arr = np.load(path)
    else:
        arr = _load_text_matrix(path)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size < 2:
            raise ValueError("EtCO2 trace must contain at least 2 samples.")
        return arr.astype(np.float32), None

    if arr.ndim != 2:
        raise ValueError(f"Unsupported EtCO2 array shape {arr.shape}; expected 1D or 2D.")

    finite_rows = np.all(np.isfinite(arr), axis=1)
    arr = arr[finite_rows]
    if arr.shape[0] < 2:
        raise ValueError("EtCO2 file must contain at least 2 finite rows.")

    n_cols = arr.shape[1]
    if n_cols == 1:
        return arr[:, 0].astype(np.float32), None

    if time_column < 0 or time_column >= n_cols or value_column < 0 or value_column >= n_cols:
        raise ValueError(
            f"Requested EtCO2 columns time={time_column}, value={value_column}, "
            f"but file only has {n_cols} column(s)."
        )

    times_sec = arr[:, time_column].astype(np.float32)
    values = arr[:, value_column].astype(np.float32)

    order = np.argsort(times_sec)
    times_sec = times_sec[order]
    values = values[order]

    keep = np.ones(times_sec.shape[0], dtype=bool)
    keep[1:] = np.diff(times_sec) > 0
    times_sec = times_sec[keep]
    values = values[keep]

    if times_sec.shape[0] < 2:
        raise ValueError("EtCO2 time column must contain at least 2 unique ascending samples.")

    return values.astype(np.float32), times_sec.astype(np.float32)


def resample_signal_to_target_times(signal, source_times_sec, target_times_sec):
    if signal.ndim != 1 or source_times_sec.ndim != 1 or target_times_sec.ndim != 1:
        raise ValueError("signal, source_times_sec, and target_times_sec must be 1D.")
    if signal.size != source_times_sec.size:
        raise ValueError("signal and source_times_sec must have the same length.")
    f = interp1d(
        source_times_sec,
        signal,
        kind="linear",
        bounds_error=False,
        fill_value=(float(signal[0]), float(signal[-1])),
    )
    return np.asarray(f(target_times_sec), dtype=np.float32)


def central_slice_index(shape_3d):
    return int(shape_3d[2] // 2)


def normalize_to_m11(x, eps=1e-8):
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax - xmin < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (2.0 * (x - xmin) / (xmax - xmin) - 1.0).astype(np.float32)


def compute_percent_bold_change(bold_4d, baseline_vols):
    if bold_4d.ndim != 4:
        raise ValueError("BOLD must be 4D.")
    if baseline_vols < 1 or baseline_vols > bold_4d.shape[-1]:
        raise ValueError("baseline_vols must be within [1, T]")

    s0 = np.mean(bold_4d[..., :baseline_vols], axis=-1)
    s0_safe = np.where(np.abs(s0) < 1e-6, 1e-6, s0)
    psc = 100.0 * (bold_4d - s0_safe[..., None]) / s0_safe[..., None]
    return psc.astype(np.float32), s0.astype(np.float32)


def build_slice_mask(slice_2d_t, mask_2d, std_thr=1e-6):
    if mask_2d is not None:
        m = mask_2d > 0
    else:
        m = np.std(slice_2d_t, axis=-1) > std_thr
    return m.astype(bool)


def resolve_bold_paths(bold_arg):
    if bold_arg.is_file():
        return [bold_arg]
    if bold_arg.is_dir():
        paths = sorted(list(bold_arg.rglob("*.nii")) + list(bold_arg.rglob("*.nii.gz")))
        if not paths:
            raise FileNotFoundError(f"No NIfTI files found under directory: {bold_arg}")
        return paths
    raise FileNotFoundError(f"--bold path does not exist: {bold_arg}")


def select_bold_paths_for_adaptation(bold_paths_all, split_json, ignore_split_json):
    if ignore_split_json or split_json is None:
        print(f"Found {len(bold_paths_all)} BOLD file(s); adapting all provided inputs.")
        return bold_paths_all

    with open(split_json, "r") as f:
        split_info = json.load(f)

    if "unused_bold_paths" not in split_info:
        raise ValueError("split_json file must contain 'unused_bold_paths' for adaptation selection.")

    unseen_paths = [Path(p).resolve() for p in split_info["unused_bold_paths"]]
    unseen_set = {str(p) for p in unseen_paths}

    bold_paths = [p for p in bold_paths_all if str(p.resolve()) in unseen_set]

    matched_by_name = 0
    if len(bold_paths) == 0 and len(bold_paths_all) > 0:
        unseen_by_name: Dict[str, List[Path]] = {}
        for p in unseen_paths:
            unseen_by_name.setdefault(p.name, []).append(p)

        filename_matches: List[Path] = []
        ambiguous_names: List[str] = []
        for p in bold_paths_all:
            matches = unseen_by_name.get(p.name, [])
            if len(matches) == 1:
                filename_matches.append(p)
            elif len(matches) > 1:
                ambiguous_names.append(p.name)

        if filename_matches:
            bold_paths = filename_matches
            matched_by_name = len(filename_matches)
            print(
                "No exact path matches were found in the unseen set; "
                "falling back to filename matching."
            )
        if ambiguous_names:
            unique_ambiguous = sorted(set(ambiguous_names))
            print(
                "Warning: Some BOLD files had ambiguous filename matches in split_json "
                f"and were skipped: {unique_ambiguous}"
            )

    print(
        f"Found {len(bold_paths_all)} BOLD files, {len(bold_paths)} of which are in the unseen set for adaptation."
    )
    if matched_by_name:
        print(f"Matched {matched_by_name} adaptation file(s) by filename fallback.")

    if len(bold_paths) == 0:
        sample_bold = str(bold_paths_all[0].resolve()) if bold_paths_all else "<none>"
        sample_unseen = str(unseen_paths[0]) if unseen_paths else "<none>"
        print("No BOLD files found for adaptation. Please check --bold path and split JSON.")
        print(f"Example resolved BOLD path: {sample_bold}")
        print(f"Example unseen path from split JSON: {sample_unseen}")

    missing = [p for p in split_info["unused_bold_paths"] if not Path(p).exists()]
    if missing:
        print("Warning: The following paths listed in split_json were not found and will be skipped:")
        for p in missing:
            print(f"  {p}")

    return bold_paths


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=20.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = np.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SirenNet(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True, first_omega_0=20.0, hidden_omega_0=20.0):
        super().__init__()

        layers = [
            SineLayer(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega_0,
            )
        ]

        for _ in range(hidden_layers):
            layers.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = np.sqrt(6.0 / hidden_features) / hidden_omega_0
                final_linear.weight.uniform_(-bound, bound)
                if final_linear.bias is not None:
                    final_linear.bias.uniform_(-bound, bound)
            layers.append(final_linear)
        else:
            layers.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TissueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = SirenNet(
            in_features=3,
            hidden_features=192,
            hidden_layers=3,
            out_features=1,
            outermost_linear=True,
            first_omega_0=20.0,
            hidden_omega_0=20.0,
        )

    def forward(self, txy):
        return self.net(txy)


class ParamNet(nn.Module):
    def __init__(self, cvr_bounds, delay_bounds):
        super().__init__()
        self.net = SirenNet(
            in_features=2,
            hidden_features=96,
            hidden_layers=2,
            out_features=2,
            outermost_linear=True,
            first_omega_0=20.0,
            hidden_omega_0=20.0,
        )
        self.cvr_min, self.cvr_max = cvr_bounds
        self.delay_min, self.delay_max = delay_bounds

    def forward(self, xy):
        raw = self.net(xy)
        cvr_raw = raw[:, 0:1]
        delay_raw = raw[:, 1:2]

        cvr = self.cvr_min + (self.cvr_max - self.cvr_min) * torch.sigmoid(cvr_raw)
        delay = self.delay_min + (self.delay_max - self.delay_min) * torch.sigmoid(delay_raw)
        return cvr, delay


class TorchLinearInterpolator1D(nn.Module):
    def __init__(self, times_sec, values):
        super().__init__()
        if times_sec.ndim != 1 or values.ndim != 1 or len(times_sec) != len(values):
            raise ValueError("times_sec and values must be 1D arrays of same length")
        self.register_buffer("times", torch.tensor(times_sec.astype(np.float32)))
        self.register_buffer("values", torch.tensor(values.astype(np.float32)))

    def forward(self, query_t):
        q = query_t.reshape(-1)
        t = self.times
        v = self.values

        q_clamped = torch.clamp(q, float(t[0]), float(t[-1]))
        idx = torch.searchsorted(t, q_clamped, right=True)
        idx = torch.clamp(idx, 1, len(t) - 1)

        t0 = t[idx - 1]
        t1 = t[idx]
        v0 = v[idx - 1]
        v1 = v[idx]

        w = (q_clamped - t0) / torch.clamp(t1 - t0, min=1e-8)
        out = v0 + w * (v1 - v0)
        return out.reshape(query_t.shape)


def prepare_single_subject_data(bold_path, mask_3d, tr, baseline_vols, slice_idx=None):
    bold_img = load_nifti(bold_path)
    bold = get_data(bold_img)

    if bold.ndim != 4:
        raise ValueError(f"BOLD must be 4D, got shape {bold.shape}")

    X, Y, Z, T = bold.shape

    if slice_idx is None:
        slice_idx = central_slice_index((X, Y, Z))

    bold_psc, _ = compute_percent_bold_change(bold, baseline_vols=baseline_vols)
    sl = bold_psc[:, :, slice_idx, :]

    mask2d = None if mask_3d is None else mask_3d[:, :, slice_idx]
    full_mask2d = build_slice_mask(sl, mask2d)

    yy, xx = np.meshgrid(np.arange(Y), np.arange(X))
    x_norm_img = normalize_to_m11(xx.astype(np.float32))
    y_norm_img = normalize_to_m11(yy.astype(np.float32))

    x_vox = x_norm_img[full_mask2d]
    y_vox = y_norm_img[full_mask2d]
    yobs = sl[full_mask2d, :]

    t_sec = np.arange(T, dtype=np.float32) * float(tr)
    t_norm = normalize_to_m11(t_sec)
    tmax = float(max(t_sec[-1], 1e-6))

    data = {
        "slice_idx": np.array(slice_idx, dtype=np.int32),
        "full_mask2d": full_mask2d.astype(np.uint8),
        "train_mask2d": full_mask2d.astype(np.uint8),
        "x_vox": x_vox.astype(np.float32),
        "y_vox": y_vox.astype(np.float32),
        "yobs_vox_t": yobs[None, ...].astype(np.float32),
        "t_sec": t_sec.astype(np.float32),
        "t_norm": t_norm.astype(np.float32),
        "tmax": np.array(tmax, dtype=np.float32),
        "shape_slice": np.array([X, Y], dtype=np.int32),
        "n_vox": np.array(x_vox.shape[0], dtype=np.int32),
        "n_vox_all": np.array(x_vox.shape[0], dtype=np.int32),
        "T": np.array(T, dtype=np.int32),
        "n_datasets": np.array(1, dtype=np.int32),
        "bold_paths": np.array([str(bold_path)], dtype=object),
    }
    return data, bold_img


def bold_cvr_ode_rhs(y_hat, cvr_mag, delay_sec, t_sec, etco2_interp, etco2_baseline, ode_T):
    et_shift = etco2_interp(t_sec - delay_sec)
    drive = cvr_mag * (et_shift - etco2_baseline)
    rhs = (drive - y_hat) / float(ode_T)
    return rhs


def adapt_pinn_to_subject(data, etco2_tr, tr, cvr_bounds, delay_bounds, ode_T, epochs, batch_size, lr, lambda_data, lambda_phys, device, pretrained_tissue, pretrained_param, freeze_tissue=False, freeze_param=False, steps_per_epoch=200, log_every=25):

    tissue_net = TissueNet().to(device)
    param_net = ParamNet(cvr_bounds=cvr_bounds, delay_bounds=delay_bounds).to(device)

    tissue_ckpt = load_checkpoint(pretrained_tissue, device)
    param_ckpt = load_checkpoint(pretrained_param, device)

    tissue_state = tissue_ckpt["model_state_dict"] if isinstance(tissue_ckpt, dict) and "model_state_dict" in tissue_ckpt else tissue_ckpt
    param_state = param_ckpt["model_state_dict"] if isinstance(param_ckpt, dict) and "model_state_dict" in param_ckpt else param_ckpt

    tissue_net.load_state_dict(tissue_state)
    param_net.load_state_dict(param_state)

    if freeze_tissue:
        for p in tissue_net.parameters():
            p.requires_grad = False

    if freeze_param:
        for p in param_net.parameters():
            p.requires_grad = False

    params = [p for p in list(tissue_net.parameters()) + list(param_net.parameters()) if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    x_vox = torch.tensor(data["x_vox"], dtype=torch.float32, device=device)
    y_vox = torch.tensor(data["y_vox"], dtype=torch.float32, device=device)
    yobs_all = torch.tensor(data["yobs_vox_t"], dtype=torch.float32, device=device)
    t_norm_all = torch.tensor(data["t_norm"], dtype=torch.float32, device=device)
    t_sec_all = torch.tensor(data["t_sec"], dtype=torch.float32, device=device)

    n_ds = int(data["n_datasets"])
    n_vox = int(data["n_vox"])
    T = int(data["T"])

    t_sec_np = data["t_sec"].astype(np.float32)
    tmax = float(data["tmax"])
    dt_norm_dt_sec = 2.0 / max(tmax, 1e-6)

    etco2_baseline = float(np.mean(etco2_tr[:max(1, min(30, len(etco2_tr)))]))
    etco2_interp = TorchLinearInterpolator1D(t_sec_np, etco2_tr.astype(np.float32)).to(device)

    history = {"loss": [], "data": [], "phys": []}
    best_loss = float("inf")
    best_state = None
    best_epoch = -1

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        total_data = 0.0
        total_phys = 0.0

        for _ in range(steps_per_epoch):
            ds_idx = torch.randint(0, n_ds, (batch_size,), device=device)
            vox_idx = torch.randint(0, n_vox, (batch_size,), device=device)
            time_idx = torch.randint(0, T, (batch_size,), device=device)

            xb = x_vox[vox_idx].unsqueeze(1)
            yb = y_vox[vox_idx].unsqueeze(1)
            tb_norm = t_norm_all[time_idx].unsqueeze(1).detach().requires_grad_(True)
            tb_sec = t_sec_all[time_idx].unsqueeze(1)
            sb = yobs_all[ds_idx, vox_idx, time_idx].unsqueeze(1)

            txy = torch.cat([tb_norm, xb, yb], dim=1)
            xy = torch.cat([xb, yb], dim=1)

            y_hat = tissue_net(txy)
            cvr_hat, delay_hat = param_net(xy)

            dy_dt_norm = torch.autograd.grad(
                outputs=y_hat,
                inputs=tb_norm,
                grad_outputs=torch.ones_like(y_hat),
                create_graph=True,
                only_inputs=True,
            )[0]

            dy_dt_sec = dy_dt_norm * dt_norm_dt_sec

            rhs = bold_cvr_ode_rhs(
                y_hat=y_hat,
                cvr_mag=cvr_hat,
                delay_sec=delay_hat,
                t_sec=tb_sec,
                etco2_interp=etco2_interp,
                etco2_baseline=etco2_baseline,
                ode_T=ode_T,
            )

            loss_data = F.mse_loss(y_hat, sb)
            loss_phys = F.mse_loss(dy_dt_sec, rhs)
            loss = lambda_data * loss_data + lambda_phys * loss_phys

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            total_data += float(loss_data.item())
            total_phys += float(loss_phys.item())

        mean_loss = total_loss / steps_per_epoch
        mean_data = total_data / steps_per_epoch
        mean_phys = total_phys / steps_per_epoch

        history["loss"].append(mean_loss)
        history["data"].append(mean_data)
        history["phys"].append(mean_phys)

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = ep
            best_state = {
                "tissue": {k: v.detach().cpu().clone() for k, v in tissue_net.state_dict().items()},
                "param": {k: v.detach().cpu().clone() for k, v in param_net.state_dict().items()},
            }

        if ep % log_every == 0 or ep == 1 or ep == epochs:
            print(
                f"[adapt epoch {ep:4d}/{epochs}] "
                f"loss={mean_loss:.6e} data={mean_data:.6e} phys={mean_phys:.6e}",
                flush=True,
            )

    if best_state is not None:
        tissue_net.load_state_dict(best_state["tissue"])
        param_net.load_state_dict(best_state["param"])

    best_info = {"best_epoch": best_epoch, "best_loss": best_loss}
    return tissue_net, param_net, history, best_info


@torch.no_grad()
def infer_param_maps(param_net, data, device):
    X, Y = data["shape_slice"]
    full_mask2d = data["full_mask2d"].astype(bool)

    yy, xx = np.meshgrid(np.arange(Y), np.arange(X))
    x_norm = normalize_to_m11(xx.astype(np.float32))
    y_norm = normalize_to_m11(yy.astype(np.float32))

    xy = np.stack([x_norm.reshape(-1), y_norm.reshape(-1)], axis=1)
    xy_t = torch.tensor(xy, dtype=torch.float32, device=device)
    cvr_hat, delay_hat = param_net(xy_t)

    cvr_map = cvr_hat.cpu().numpy().reshape(X, Y)
    delay_map = delay_hat.cpu().numpy().reshape(X, Y)

    cvr_map[~full_mask2d] = 0.0
    delay_map[~full_mask2d] = 0.0
    return cvr_map.astype(np.float32), delay_map.astype(np.float32)


@torch.no_grad()
def reconstruct_signal_slice(tissue_net, data, device):
    X, Y = data["shape_slice"]
    T = int(data["T"])

    yy, xx = np.meshgrid(np.arange(Y), np.arange(X))
    x_norm = normalize_to_m11(xx.astype(np.float32))
    y_norm = normalize_to_m11(yy.astype(np.float32))
    t_norm = data["t_norm"]
    full_mask2d = data["full_mask2d"].astype(bool)

    out = np.zeros((X, Y, T), dtype=np.float32)
    for ti in range(T):
        t_plane = np.full((X, Y), t_norm[ti], dtype=np.float32)
        txy = np.stack(
            [t_plane.reshape(-1), x_norm.reshape(-1), y_norm.reshape(-1)],
            axis=1,
        )
        txy_t = torch.tensor(txy, dtype=torch.float32, device=device)
        pred = tissue_net(txy_t).cpu().numpy().reshape(X, Y)
        pred[~full_mask2d] = 0.0
        out[..., ti] = pred
    return out


def main():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    p = argparse.ArgumentParser(description="Adapt pretrained BOLD PINN checkpoints to unseen BOLD images.")
    p.add_argument("--bold", type=Path, default=Path("/workspace/tCNR_5.000"), help="Single unseen BOLD file or directory of unseen BOLD files")
    p.add_argument("--outdir", type=Path, default=Path("/workspace/pinn_tcnr5.0_40_f"), help="Output root directory")
    p.add_argument("--pretrained-tissue", type=Path, default=Path("/workspace/pinn_tcnr5.0_60_f/best_tissue_net.pt"))
    p.add_argument("--pretrained-param", type=Path, default=Path("/workspace/pinn_tcnr5.0_60_f/best_param_net.pt"))

    p.add_argument("--mask", type=Path, default=None)
    p.add_argument("--slice-index", type=int, default=62)
    p.add_argument("--tr", type=float, default=1.55)
    p.add_argument("--baseline-vols", type=int, default=30)

    p.add_argument("--extra-pre", type=int, default=31)
    p.add_argument("--extra-post", type=int, default=93)
    p.add_argument("--etco2-sample-dt", type=float, default=1.0)
    p.add_argument("--etco2-file", type=Path, default=None, help="Optional real EtCO2 trace (.txt, .csv, or .npy)")
    p.add_argument("--etco2-time-column", type=int, default=0, help="Time column for tabular EtCO2 input")
    p.add_argument("--etco2-value-column", type=int, default=1, help="Value column for tabular EtCO2 input")

    p.add_argument("--adapt-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=12248)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-data", type=float, default=1.0)
    p.add_argument("--lambda-phys", type=float, default=0.5)
    p.add_argument("--steps-per-epoch", type=int, default=1000)

    p.add_argument("--ode-T", type=float, default=15.0)
    p.add_argument("--cvr-min", type=float, default=-0.4)
    p.add_argument("--cvr-max", type=float, default=1.8)
    p.add_argument("--delay-min", type=float, default=0.0)
    p.add_argument("--delay-max", type=float, default=60.0)

    p.add_argument("--freeze-tissue", action="store_true")
    p.add_argument("--freeze-param", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--split-json", type=Path, default=Path("/workspace/pinn_tcnr5.0_60_f/data_split.json"))
    p.add_argument("--ignore-split-json", action="store_true", help="Adapt all files matched by --bold instead of filtering via split_json")

    args = p.parse_args()
    set_seed(args.seed)

    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    bold_paths_all = resolve_bold_paths(args.bold)
    bold_paths = select_bold_paths_for_adaptation(
        bold_paths_all=bold_paths_all,
        split_json=None if args.ignore_split_json else args.split_json,
        ignore_split_json=args.ignore_split_json,
    )
    if len(bold_paths) == 0:
        return

    if len(bold_paths) > 1:
        print(
            f"Multiple BOLD files matched ({len(bold_paths)} total). "
            "All matched files will be adapted."
        )

    mask = None
    if args.mask is not None:
        mask = get_data(load_nifti(args.mask))

    target_etco2_times_sec = None
    if args.etco2_file is not None:
        etco2_raw, target_etco2_times_sec = load_etco2_from_file(
            args.etco2_file,
            sample_dt=args.etco2_sample_dt,
            time_column=args.etco2_time_column,
            value_column=args.etco2_value_column,
        )
        print(f"Loaded EtCO2 trace from {args.etco2_file}")
    else:
        etco2_raw = build_long_etco2(extra_pre=args.extra_pre, extra_post=args.extra_post)
        print("Using built-in simulated EtCO2 trace.")

    summary = []

    for i, bold_path in enumerate(bold_paths, start=1):
        print("=" * 80)
        print(f"[{i}/{len(bold_paths)}] Adapting to {bold_path}")
        print("=" * 80)

        data, bold_img = prepare_single_subject_data(
            bold_path=bold_path,
            mask_3d=mask,
            tr=args.tr,
            baseline_vols=args.baseline_vols,
            slice_idx=args.slice_index,
        )

        if target_etco2_times_sec is None:
            etco2_tr = resample_signal_to_tr(
                signal=etco2_raw,
                signal_dt=args.etco2_sample_dt,
                n_time=int(data["T"]),
                tr=args.tr,
            )
        else:
            etco2_tr = resample_signal_to_target_times(
                signal=etco2_raw,
                source_times_sec=target_etco2_times_sec,
                target_times_sec=data["t_sec"],
            )

        tissue_net, param_net, history, best_info = adapt_pinn_to_subject(
            data=data,
            etco2_tr=etco2_tr,
            tr=args.tr,
            cvr_bounds=(args.cvr_min, args.cvr_max),
            delay_bounds=(args.delay_min, args.delay_max),
            ode_T=args.ode_T,
            epochs=args.adapt_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_data=args.lambda_data,
            lambda_phys=args.lambda_phys,
            device=device,
            pretrained_tissue=args.pretrained_tissue,
            pretrained_param=args.pretrained_param,
            freeze_tissue=args.freeze_tissue,
            freeze_param=args.freeze_param,
            steps_per_epoch=args.steps_per_epoch,
        )

        cvr_map_2d, delay_map_2d = infer_param_maps(param_net, data, device=device)
        recon_slice = reconstruct_signal_slice(tissue_net, data, device=device)

        ref_bold = get_data(bold_img)
        X, Y, Z, T = ref_bold.shape
        slice_idx = int(data["slice_idx"])

        cvr_3d = np.zeros((X, Y, Z), dtype=np.float32)
        delay_3d = np.zeros((X, Y, Z), dtype=np.float32)
        recon_4d = np.zeros((X, Y, Z, T), dtype=np.float32)
        full_mask_3d = np.zeros((X, Y, Z), dtype=np.uint8)

        cvr_3d[:, :, slice_idx] = cvr_map_2d
        delay_3d[:, :, slice_idx] = delay_map_2d
        recon_4d[:, :, slice_idx, :] = recon_slice
        full_mask_3d[:, :, slice_idx] = data["full_mask2d"]

        subj_name = bold_path.name.replace(".nii.gz", "").replace(".nii", "")
        subj_out = args.outdir / subj_name
        subj_out.mkdir(parents=True, exist_ok=True)

        save_nifti_like(cvr_3d, bold_img, subj_out / "pinn_cvr_mag.nii.gz")
        save_nifti_like(delay_3d, bold_img, subj_out / "pinn_delay.nii.gz")
        save_nifti_like(recon_4d, bold_img, subj_out / "pinn_recon_psc.nii.gz")
        save_nifti_like(full_mask_3d.astype(np.float32), bold_img, subj_out / "pinn_slice_mask_full.nii.gz")

        torch.save({"model_state_dict": tissue_net.state_dict(), "best_info": best_info}, subj_out / "adapted_tissue_net.pt")
        torch.save({"model_state_dict": param_net.state_dict(), "best_info": best_info}, subj_out / "adapted_param_net.pt")

        with open(subj_out / "adapt_history.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(subj_out / "adapt_best_info.json", "w") as f:
            json.dump(best_info, f, indent=2)
        with open(subj_out / "adapt_config.json", "w") as f:
            json.dump(vars(args), f, indent=2, default=str)

        summary.append({
            "bold_path": str(bold_path),
            "output_dir": str(subj_out),
            "best_epoch": int(best_info["best_epoch"]),
            "best_loss": float(best_info["best_loss"]),
        })

    with open(args.outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()

