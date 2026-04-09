
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
from scipy.optimize import least_squares

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


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
            hidden_features=128,
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
            hidden_features=64,
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


def prepare_training_data(bold_psc_4d, mask_3d, tr, slice_idx=None, max_train_voxels=None, seed=0):

    X, Y, Z, T = bold_psc_4d.shape

    if slice_idx is None:
        slice_idx = central_slice_index((X, Y, Z))

    sl = bold_psc_4d[:, :, slice_idx, :]
    mask2d = None if mask_3d is None else mask_3d[:, :, slice_idx]
    mask2d = build_slice_mask(sl, mask2d)

    yy, xx = np.meshgrid(np.arange(Y), np.arange(X))
    x_norm_img = normalize_to_m11(xx.astype(np.float32))
    y_norm_img = normalize_to_m11(yy.astype(np.float32))

    t_sec = np.arange(T, dtype=np.float32) * float(tr)
    t_norm = normalize_to_m11(t_sec)
    tmax = float(max(t_sec[-1], 1e-6))

    vox_idx = np.argwhere(mask2d)
    n_vox_all = vox_idx.shape[0]
    if n_vox_all == 0:
        raise RuntimeError("Mask on selected slice is empty.")

    if max_train_voxels is not None and max_train_voxels < n_vox_all:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(n_vox_all, size=max_train_voxels, replace=False)
        train_vox_idx = vox_idx[chosen]
    else:
        train_vox_idx = vox_idx

    train_mask2d = np.zeros_like(mask2d, dtype=bool)
    train_mask2d[train_vox_idx[:, 0], train_vox_idx[:, 1]] = True

    x_vox = x_norm_img[train_mask2d]
    y_vox = y_norm_img[train_mask2d]
    yobs = sl[train_mask2d, :]

    n_vox = x_vox.shape[0]

    x_rep = np.repeat(x_vox[:, None], T, axis=1).reshape(-1, 1)
    y_rep = np.repeat(y_vox[:, None], T, axis=1).reshape(-1, 1)
    t_rep_norm = np.repeat(t_norm[None, :], n_vox, axis=0).reshape(-1, 1)
    t_rep_sec = np.repeat(t_sec[None, :], n_vox, axis=0).reshape(-1, 1)
    signal_rep = yobs.reshape(-1, 1)

    return {
        "slice_idx": np.array(slice_idx, dtype=np.int32),
        "full_mask2d": mask2d.astype(np.uint8),
        "train_mask2d": train_mask2d.astype(np.uint8),
        "x_vox": x_vox.astype(np.float32),
        "y_vox": y_vox.astype(np.float32),
        "yobs_vox_t": yobs.astype(np.float32),
        "x_flat": x_rep.astype(np.float32),
        "y_flat": y_rep.astype(np.float32),
        "t_flat_norm": t_rep_norm.astype(np.float32),
        "t_flat_sec": t_rep_sec.astype(np.float32),
        "signal_flat": signal_rep.astype(np.float32),
        "t_sec": t_sec.astype(np.float32),
        "t_norm": t_norm.astype(np.float32),
        "tmax": np.array(tmax, dtype=np.float32),
        "shape_slice": np.array([X, Y], dtype=np.int32),
        "n_vox": np.array(n_vox, dtype=np.int32),
        "n_vox_all": np.array(n_vox_all, dtype=np.int32),
        "T": np.array(T, dtype=np.int32),
    }


def prepare_multi_training_data(bold_paths, mask_3d, tr, baseline_vols, slice_idx=None, max_train_voxels=None, seed=0):

    rng = np.random.default_rng(seed)

    ref_img = load_nifti(bold_paths[0])
    ref_bold = get_data(ref_img)
    if ref_bold.ndim != 4:
        raise ValueError(f"BOLD must be 4D, got shape {ref_bold.shape}")

    X, Y, Z, T = ref_bold.shape

    if slice_idx is None:
        slice_idx = central_slice_index((X, Y, Z))

    yy, xx = np.meshgrid(np.arange(Y), np.arange(X))
    x_norm_img = normalize_to_m11(xx.astype(np.float32))
    y_norm_img = normalize_to_m11(yy.astype(np.float32))

    t_sec = np.arange(T, dtype=np.float32) * float(tr)
    t_norm = normalize_to_m11(t_sec)
    tmax = float(max(t_sec[-1], 1e-6))

    ref_psc, _ = compute_percent_bold_change(ref_bold, baseline_vols=baseline_vols)
    ref_slice = ref_psc[:, :, slice_idx, :]
    mask2d = None if mask_3d is None else mask_3d[:, :, slice_idx]
    full_mask2d = build_slice_mask(ref_slice, mask2d)

    vox_idx = np.argwhere(full_mask2d)
    n_vox_all = vox_idx.shape[0]
    if n_vox_all == 0:
        raise RuntimeError("Mask on selected slice is empty.")

    if max_train_voxels is not None and max_train_voxels < n_vox_all:
        chosen = rng.choice(n_vox_all, size=max_train_voxels, replace=False)
        train_vox_idx = vox_idx[chosen]
    else:
        train_vox_idx = vox_idx

    train_mask2d = np.zeros_like(full_mask2d, dtype=bool)
    train_mask2d[train_vox_idx[:, 0], train_vox_idx[:, 1]] = True

    x_vox = x_norm_img[train_mask2d]
    y_vox = y_norm_img[train_mask2d]
    n_vox = x_vox.shape[0]
    n_ds = len(bold_paths)

    yobs_all = np.zeros((n_ds, n_vox, T), dtype=np.float32)

    for dsi, path in enumerate(bold_paths):
        print(f"[prepare_multi_training_data] Loading dataset {dsi+1}/{n_ds}: {path.name}", flush=True)

        img = load_nifti(path)
        bold = get_data(img)

        if bold.shape != (X, Y, Z, T):
            raise ValueError(
                f"All BOLD datasets must have same shape. "
                f"Expected {(X, Y, Z, T)}, got {bold.shape} for {path}"
            )

        bold_psc, _ = compute_percent_bold_change(bold, baseline_vols=baseline_vols)
        sl = bold_psc[:, :, slice_idx, :]
        yobs_all[dsi] = sl[train_mask2d, :]

    data = {
        "slice_idx": np.array(slice_idx, dtype=np.int32),
        "full_mask2d": full_mask2d.astype(np.uint8),
        "train_mask2d": train_mask2d.astype(np.uint8),
        "x_vox": x_vox.astype(np.float32),
        "y_vox": y_vox.astype(np.float32),
        "yobs_vox_t": yobs_all.astype(np.float32),
        "t_sec": t_sec.astype(np.float32),
        "t_norm": t_norm.astype(np.float32),
        "tmax": np.array(tmax, dtype=np.float32),
        "shape_slice": np.array([X, Y], dtype=np.int32),
        "n_vox": np.array(n_vox, dtype=np.int32),
        "n_vox_all": np.array(n_vox_all, dtype=np.int32),
        "T": np.array(T, dtype=np.int32),
        "n_datasets": np.array(n_ds, dtype=np.int32),
        "bold_paths": np.array([str(p) for p in bold_paths], dtype=object),
    }
    return data, ref_img


def bold_cvr_ode_rhs(y_hat, cvr_mag, delay_sec, t_sec, etco2_interp, etco2_baseline, ode_T):

    et_shift = etco2_interp(t_sec - delay_sec)
    drive = cvr_mag * (et_shift - etco2_baseline)
    rhs = (drive - y_hat) / float(ode_T)
    return rhs


def train_pinn(data, etco2_tr, tr, cvr_bounds, delay_bounds, ode_T, epochs, warmup_epochs, batch_size, lr, lambda_data, lambda_phys, device, log_every=50, pretrained_tissue=None, pretrained_param=None, freeze_tissue=False, freeze_param=False, steps_per_epoch=200, dataset_reduction="mean", debug_batch_shapes=False):

    tissue_net = TissueNet().to(device)
    param_net = ParamNet(cvr_bounds=cvr_bounds, delay_bounds=delay_bounds).to(device)

    if pretrained_tissue is not None:
        ckpt = torch.load(pretrained_tissue, map_location=device)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt and isinstance(ckpt, dict) else ckpt
        tissue_net.load_state_dict(state)
        print(f"Loaded pretrained TissueNet from {pretrained_tissue}")

    if pretrained_param is not None:
        ckpt = torch.load(pretrained_param, map_location=device)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt and isinstance(ckpt, dict) else ckpt
        param_net.load_state_dict(state)
        print(f"Loaded pretrained ParamNet from {pretrained_param}")

    if freeze_tissue:
        for param in tissue_net.parameters():
            param.requires_grad = False
        print("Froze TissueNet parameters.")

    if freeze_param:
        for param in param_net.parameters():
            param.requires_grad = False
        print("Froze ParamNet parameters.")

    trainable_params = [p for p in list(tissue_net.parameters()) + list(param_net.parameters()) if p.requires_grad]
    opt = torch.optim.Adam(trainable_params, lr=lr)

    x_vox = torch.tensor(data["x_vox"], dtype=torch.float32, device=device)
    y_vox = torch.tensor(data["y_vox"], dtype=torch.float32, device=device)
    yobs_all = torch.tensor(data["yobs_vox_t"], dtype=torch.float32, device=device)
    yobs_mean = yobs_all.mean(dim=0)
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

    history = {
        "loss": [],
        "data": [],
        "phys": [],
        "phys_weight": [],
        "epoch_time_sec": [],
    }

    if dataset_reduction not in {"mean", "sample"}:
        raise ValueError("dataset_reduction must be one of: 'mean', 'sample'")

    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    printed_debug_shapes = False

    for ep in range(1, epochs + 1):
        ep_t0 = time.time()

        total_loss_ep = 0.0
        total_data_ep = 0.0
        total_phys_ep = 0.0
        n_batches = 0

        phys_weight = 0.0 if ep <= warmup_epochs else lambda_phys

        for _ in range(steps_per_epoch):
            vox_idx = torch.randint(0, n_vox, (batch_size,), device=device)
            time_idx = torch.randint(0, T, (batch_size,), device=device)

            xb = x_vox[vox_idx].unsqueeze(1)
            yb = y_vox[vox_idx].unsqueeze(1)
            tb_norm = t_norm_all[time_idx].unsqueeze(1).detach().requires_grad_(True)
            tb_sec = t_sec_all[time_idx].unsqueeze(1)
            if dataset_reduction == "mean":
                sb = yobs_mean[vox_idx, time_idx].unsqueeze(1)
            else:
                ds_idx = torch.randint(0, n_ds, (batch_size,), device=device)
                sb = yobs_all[ds_idx, vox_idx, time_idx].unsqueeze(1)

            txy = torch.cat([tb_norm, xb, yb], dim=1)
            xy = torch.cat([xb, yb], dim=1)

            if debug_batch_shapes and not printed_debug_shapes:
                print(
                    "[debug] batch shapes: "
                    f"xb={tuple(xb.shape)} "
                    f"yb={tuple(yb.shape)} "
                    f"tb_norm={tuple(tb_norm.shape)} "
                    f"tb_sec={tuple(tb_sec.shape)} "
                    f"txy={tuple(txy.shape)} "
                    f"xy={tuple(xy.shape)} "
                    f"sb={tuple(sb.shape)}"
                )
                printed_debug_shapes = True

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
            loss = lambda_data * loss_data + phys_weight * loss_phys

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss_ep += float(loss.item())
            total_data_ep += float(loss_data.item())
            total_phys_ep += float(loss_phys.item())
            n_batches += 1

        ep_t1 = time.time()

        mean_loss = total_loss_ep / max(n_batches, 1)
        mean_data = total_data_ep / max(n_batches, 1)
        mean_phys = total_phys_ep / max(n_batches, 1)

        history["loss"].append(mean_loss)
        history["data"].append(mean_data)
        history["phys"].append(mean_phys)
        history["phys_weight"].append(float(phys_weight))
        history["epoch_time_sec"].append(float(ep_t1 - ep_t0))

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = ep
            best_state = {
                "tissue_net": {k: v.detach().cpu().clone() for k, v in tissue_net.state_dict().items()},
                "param_net": {k: v.detach().cpu().clone() for k, v in param_net.state_dict().items()},
            }

        if ep % log_every == 0 or ep == 1 or ep == epochs:
            print(
                f"[Epoch {ep:6d}/{epochs}] "
                f"loss={mean_loss:.6e} "
                f"data={mean_data:.6e} "
                f"phys={mean_phys:.6e} "
                f"phys_w={phys_weight:.3e} "
                f"epoch_time={history['epoch_time_sec'][-1]:.3f}s"
            )

    if best_state is not None:
        tissue_net.load_state_dict(best_state["tissue_net"])
        param_net.load_state_dict(best_state["param_net"])

    best_info = {
        "best_epoch": best_epoch,
        "best_loss": best_loss,
    }
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
            [
                t_plane.reshape(-1),
                x_norm.reshape(-1),
                y_norm.reshape(-1),
            ],
            axis=1,
        )
        txy_t = torch.tensor(txy, dtype=torch.float32, device=device)
        pred = tissue_net(txy_t).cpu().numpy().reshape(X, Y)
        pred[~full_mask2d] = 0.0
        out[..., ti] = pred
    return out


def simulate_ode_signal_numpy(etco2_tr, tr, cvr_mag, delay_sec, ode_T, y0=0.0):
    Tn = len(etco2_tr)
    t_sec = np.arange(Tn, dtype=np.float32) * float(tr)
    f = interp1d(
        t_sec,
        etco2_tr,
        kind="linear",
        bounds_error=False,
        fill_value=(float(etco2_tr[0]), float(etco2_tr[-1])),
    )
    et_base = float(np.mean(etco2_tr[:max(1, min(30, len(etco2_tr)))]))
    y = np.zeros(Tn, dtype=np.float32)
    y[0] = float(y0)

    for i in range(Tn - 1):
        et_shift = float(f(t_sec[i] - delay_sec))
        drive = cvr_mag * (et_shift - et_base)
        dydt = (drive - y[i]) / float(ode_T)
        y[i + 1] = y[i] + float(tr) * dydt

    return y


def fit_nlls_voxelwise(yobs_vox_t, etco2_tr, tr, cvr_bounds, delay_bounds, ode_T, lambda_delay_reg=1e-3, delay_init=None):

    N, _ = yobs_vox_t.shape
    cvr_out = np.zeros(N, dtype=np.float32)
    delay_out = np.zeros(N, dtype=np.float32)

    if delay_init is None:
        delay_init = 0.5 * (delay_bounds[0] + delay_bounds[1])

    for i in range(N):
        y = yobs_vox_t[i]
        y0 = float(y[0])

        def residual(p):
            cvr_mag, delay_sec = float(p[0]), float(p[1])
            pred = simulate_ode_signal_numpy(
                etco2_tr=etco2_tr,
                tr=tr,
                cvr_mag=cvr_mag,
                delay_sec=delay_sec,
                ode_T=ode_T,
                y0=y0,
            )
            reg = np.sqrt(lambda_delay_reg) * np.array([delay_sec - delay_init], dtype=np.float32)
            return np.concatenate([pred - y, reg], axis=0)

        x0 = np.array(
            [
                0.5 * (cvr_bounds[0] + cvr_bounds[1]),
                delay_init,
            ],
            dtype=np.float32,
        )

        res = least_squares(
            residual,
            x0=x0,
            bounds=(
                np.array([cvr_bounds[0], delay_bounds[0]], dtype=np.float32),
                np.array([cvr_bounds[1], delay_bounds[1]], dtype=np.float32),
            ),
            loss="soft_l1",
            method="trf",
            max_nfev=200,
        )
        cvr_out[i] = float(res.x[0])
        delay_out[i] = float(res.x[1])

    return cvr_out, delay_out


def pcc_masked(a, b, mask):
    aa = a[mask].astype(np.float64)
    bb = b[mask].astype(np.float64)
    if aa.size < 2:
        return np.nan
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def pre_map(pred, gt, eps=1e-6):
    denom = np.where(np.abs(gt) < eps, np.sign(gt) * eps + (gt == 0) * eps, gt)
    return 100.0 * (pred - gt) / denom


def masked_ssim(a, b, mask):
    if not HAS_SKIMAGE:
        return np.nan
    aa = a.copy()
    bb = b.copy()
    aa[~mask] = 0.0
    bb[~mask] = 0.0
    data_range = max(float(np.max(bb) - np.min(bb)), 1e-6)
    return float(ssim(aa, bb, data_range=data_range))


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device)
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        return torch.load(path, map_location=device, weights_only=False)


def load_tissue_cp(ckpt_path, device):
    model = TissueNet().to(device)
    ckpt = load_checkpoint(ckpt_path, device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    return model


def load_param_cp(ckpt_path, device, cvr_bounds, delay_bounds):
    model = ParamNet(cvr_bounds=cvr_bounds, delay_bounds=delay_bounds).to(device)
    ckpt = load_checkpoint(ckpt_path, device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return model


def main():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    p = argparse.ArgumentParser(
        description="Corrected two-network PINN for BOLD CVR on a single slice or pooled slices."
    )
    p.add_argument(
        "--bold",
        type=Path,
        default=Path("/workspace/tCNR_0.500/"),
        help="4D BOLD NIfTI OR directory containing multiple 4D BOLD NIfTIs",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("/workspace/pinn_tcnr0.5_60"),
        help="Output directory",
    )

    p.add_argument("--mask", type=Path, default=None, help="Optional 3D binary mask NIfTI")
    p.add_argument("--slice-index", type=int, default=62, help="Slice index; default is central slice")
    p.add_argument("--tr", type=float, default=1.55, help="BOLD TR in seconds")
    p.add_argument("--baseline-vols", type=int, default=30, help="Volumes used for baseline PSC conversion")

    p.add_argument(
        "--extra-pre",
        type=int,
        default=31,
        help="Extra baseline seconds prepended to built-in 1 Hz EtCO2 block paradigm",
    )
    p.add_argument(
        "--extra-post",
        type=int,
        default=93,
        help="Extra baseline seconds appended to built-in 1 Hz EtCO2 block paradigm",
    )
    p.add_argument(
        "--etco2-sample-dt",
        type=float,
        default=1.0,
        help="Sampling interval of built-in EtCO2 block paradigm in seconds",
    )

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--warmup-epochs", type=int, default=200, help="Data-only warmup epochs")
    p.add_argument("--batch-size", type=int, default=12288)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-data", type=float, default=1.0)
    p.add_argument("--lambda-phys", type=float, default=0.5)

    p.add_argument(
        "--max-train-voxels",
        type=int,
        default=2900,
        help="Subsample this many masked voxels for training; inference still runs on full slice",
    )

    p.add_argument("--ode-T", type=float, default=15.0, help="Fixed hemodynamic time constant (s)")
    p.add_argument("--cvr-min", type=float, default=-0.4, help="CVR lower bound (%/mmHg)")
    p.add_argument("--cvr-max", type=float, default=1.8, help="CVR upper bound (%/mmHg)")
    p.add_argument("--delay-min", type=float, default=0.0, help="Delay lower bound (s)")
    p.add_argument("--delay-max", type=float, default=120.0, help="Delay upper bound (s)")

    p.add_argument("--gt-cvr", type=Path, default=None, help="Optional ground-truth CVR magnitude NIfTI")
    p.add_argument("--gt-delay", type=Path, default=None, help="Optional ground-truth delay NIfTI")

    p.add_argument("--run-nlls", action="store_true", help="Run voxelwise NLLS baseline on training voxels only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--pretrained-tissue", type=Path, default=None, help="Optional pretrained TissueNet checkpoint")
    p.add_argument("--pretrained-param", type=Path, default=None, help="Optional pretrained ParamNet checkpoint")
    p.add_argument("--freeze-tissue", action="store_true", help="Freeze TissueNet weights if using pretrained checkpoint")
    p.add_argument("--freeze-param", action="store_true", help="Freeze ParamNet weights if using pretrained checkpoint")

    p.add_argument("--n-train", type=int, default=60, help="Number of training data from bold directory")

    p.add_argument("--steps-per-epoch", type=int, default=1000, help="Number of random batches per epoch (for pooled training)")
    p.add_argument(
        "--dataset-reduction",
        type=str,
        default="mean",
        choices=["mean", "sample"],
        help="How to form the data target when pooling multiple datasets",
    )
    p.add_argument(
        "--debug-batch-shapes",
        action="store_true",
        help="Print one batch of tensor shapes during training for input sanity checking",
    )

    p.add_argument("--split-json", type=Path, default=Path("/workspace/image_split.json"), help="Optional path to save JSON file with train/unused data split info")

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

    bold_paths = resolve_bold_paths(args.bold)
    print(f"Found {len(bold_paths)} BOLD dataset(s).")
    if len(bold_paths) == 0:
        raise RunTimeError(f"No BOLD files found in {args.bold}")

    if args.split_json is not None:
        with open(args.split_json, "r") as f:
            split_info = json.load(f)

        if "train_indices" in split_info:
            train_indices = split_info["train_indices"]

            if not isinstance(train_indices, list) or len(train_indices) == 0:
                raise ValueError("split JSON 'train_indices' must be a non-empty list")

            if any((not isinstance(i, int)) for i in train_indices):
                raise ValueError("All entries in 'train_indices' must be integers")

            n_bold = len(bold_paths)
            bad_idx = [i for i in train_indices if i < 0 or i >= n_bold]
            if bad_idx:
                raise ValueError(
                    f"Some train_indices are out of range for {n_bold} resolved BOLD files: {bad_idx}"
                )

            selected_paths = [bold_paths[i] for i in train_indices]
            selected_index_set = set(train_indices)
            unused_paths = [p for i, p in enumerate(bold_paths) if i not in selected_index_set]

            print(
                f"Loaded split from {args.split_json} using train_indices: "
                f"{len(selected_paths)} train, {len(unused_paths)} unused"
            )

        elif "train_bold" in split_info:
            train_names = {Path(p).name for p in split_info["train_bold"]}
            selected_paths = [p for p in bold_paths if p.name in train_names]
            unused_paths = [p for p in bold_paths if p.name not in train_names]

            if len(selected_paths) == 0:
                raise ValueError(
                    f"split JSON has 'train_bold' but none matched resolved files in {args.bold}. "
                    f"Use train_indices instead."
                )

            print(
                f"Loaded split from {args.split_json} using filename fallback: "
                f"{len(selected_paths)} train, {len(unused_paths)} unused"
            )

        else:
            raise ValueError(
                "Invalid split JSON format: need either 'train_indices' or 'train_bold'"
            )

    elif args.n_train is not None and args.n_train < len(bold_paths):
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(bold_paths))

        selected_idx = perm[:args.n_train]
        selected_paths = [bold_paths[i] for i in selected_idx]

        unused_idx = perm[args.n_train:]
        unused_paths = [bold_paths[i] for i in unused_idx]

    else:
        selected_paths = bold_paths
        unused_paths = []

    mask = None
    if args.mask is not None:
        mask_img = load_nifti(args.mask)
        mask = get_data(mask_img)

    data, bold_img = prepare_multi_training_data(
        bold_paths=selected_paths,
        mask_3d=mask,
        tr=args.tr,
        baseline_vols=args.baseline_vols,
        slice_idx=args.slice_index,
        max_train_voxels=args.max_train_voxels,
        seed=args.seed,
    )
    slice_idx = int(data["slice_idx"])

    etco2_raw = build_long_etco2(
        extra_pre=args.extra_pre,
        extra_post=args.extra_post,
    )
    etco2_tr = resample_signal_to_tr(
        signal=etco2_raw,
        signal_dt=args.etco2_sample_dt,
        n_time=int(data["T"]),
        tr=args.tr,
    )

    print(f"Built-in EtCO2 length at {args.etco2_sample_dt:.3f}s sampling: {len(etco2_raw)} samples")
    print(f"Resampled EtCO2 length at TR={args.tr:.3f}s: {len(etco2_tr)} samples")

    print(f"Training on slice index: {slice_idx}")
    print(f"Number of datasets pooled: {int(data['n_datasets'])}")
    print(f"Masked voxels on slice (full): {int(data['n_vox_all'])}")
    print(f"Masked voxels used for training per dataset: {int(data['n_vox'])}")
    total_pooled_samples = int(data["n_datasets"]) * int(data["n_vox"]) * int(data["T"])
    print(f"Total pooled samples (dataset × voxel × time): {total_pooled_samples}")

    tissue_net, param_net, history, best_info = train_pinn(
        data=data,
        etco2_tr=etco2_tr,
        tr=args.tr,
        cvr_bounds=(args.cvr_min, args.cvr_max),
        delay_bounds=(args.delay_min, args.delay_max),
        ode_T=args.ode_T,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_data=args.lambda_data,
        lambda_phys=args.lambda_phys,
        device=device,
        log_every=max(1, args.epochs // 20),
        pretrained_tissue=args.pretrained_tissue,
        pretrained_param=args.pretrained_param,
        freeze_tissue=args.freeze_tissue,
        freeze_param=args.freeze_param,
        steps_per_epoch=args.steps_per_epoch,
        dataset_reduction=args.dataset_reduction,
        debug_batch_shapes=args.debug_batch_shapes,
    )

    print(f"Best training epoch: {best_info['best_epoch']} best_loss={best_info['best_loss']:.6e}")

    train_indices = [i for i, p in enumerate(bold_paths) if p in selected_paths]
    unused_indices = [i for i, p in enumerate(bold_paths) if p in unused_paths]

    split_info = {
        "train_indices": train_indices,
        "unused_indices": unused_indices,
        "train_bold_paths": [str(p) for p in selected_paths],
        "unused_bold_paths": [str(p) for p in unused_paths],
        "n_train": len(selected_paths),
        "n_unseen": len(unused_paths),
        "seed": args.seed,
    }

    split_info["source_split_json"] = str(args.split_json) if args.split_json is not None else None

    with open(args.outdir / "data_split.json", "w") as f:
        json.dump(split_info, f, indent=2)

    cvr_map_2d, delay_map_2d = infer_param_maps(param_net, data, device=device)
    recon_slice = reconstruct_signal_slice(tissue_net, data, device=device)

    ref_bold = get_data(bold_img)
    X, Y, Z, T = ref_bold.shape
    cvr_3d = np.zeros((X, Y, Z), dtype=np.float32)
    delay_3d = np.zeros((X, Y, Z), dtype=np.float32)
    recon_4d = np.zeros((X, Y, Z, T), dtype=np.float32)
    full_mask_3d = np.zeros((X, Y, Z), dtype=np.uint8)
    train_mask_3d = np.zeros((X, Y, Z), dtype=np.uint8)

    cvr_3d[:, :, slice_idx] = cvr_map_2d
    delay_3d[:, :, slice_idx] = delay_map_2d
    recon_4d[:, :, slice_idx, :] = recon_slice
    full_mask_3d[:, :, slice_idx] = data["full_mask2d"]
    train_mask_3d[:, :, slice_idx] = data["train_mask2d"]

    save_nifti_like(cvr_3d, bold_img, args.outdir / "pinn_cvr_mag.nii.gz")
    save_nifti_like(delay_3d, bold_img, args.outdir / "pinn_delay.nii.gz")
    save_nifti_like(recon_4d, bold_img, args.outdir / "pinn_recon_psc.nii.gz")
    save_nifti_like(full_mask_3d.astype(np.float32), bold_img, args.outdir / "pinn_slice_mask_full.nii.gz")
    save_nifti_like(train_mask_3d.astype(np.float32), bold_img, args.outdir / "pinn_slice_mask_train.nii.gz")

    torch.save({
        "model_state_dict": tissue_net.state_dict(),
        "best_info": best_info,
        "config": vars(args),
        "n_datasets": int(data["n_datasets"]),
        "training_bolds": [str(p) for p in selected_paths],
        "slice_index": slice_idx,
    }, args.outdir / "best_tissue_net.pt")

    torch.save({
        "model_state_dict": param_net.state_dict(),
        "best_info": best_info,
        "config": vars(args),
        "n_datasets": int(data["n_datasets"]),
        "training_bolds": [str(p) for p in selected_paths],
        "slice_index": slice_idx,
    }, args.outdir / "best_param_net.pt")

    with open(args.outdir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(args.outdir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(args.outdir / "best_info.json", "w") as f:
        json.dump(best_info, f, indent=2)

    metrics = {}
    full_mask2d = data["full_mask2d"].astype(bool)

    if args.gt_cvr is not None:
        gt_cvr = get_data(load_nifti(args.gt_cvr))
        gt_cvr_2d = gt_cvr[:, :, slice_idx]
        metrics["cvr_pcc"] = pcc_masked(cvr_map_2d, gt_cvr_2d, full_mask2d)
        metrics["cvr_ssim"] = masked_ssim(cvr_map_2d, gt_cvr_2d, full_mask2d)
        cvr_pre = pre_map(cvr_map_2d, gt_cvr_2d)
        cvr_pre[~full_mask2d] = 0.0
        cvr_pre_3d = np.zeros((X, Y, Z), dtype=np.float32)
        cvr_pre_3d[:, :, slice_idx] = cvr_pre
        save_nifti_like(cvr_pre_3d, bold_img, args.outdir / "pinn_cvr_pre.nii.gz")

    if args.gt_delay is not None:
        gt_delay = get_data(load_nifti(args.gt_delay))
        gt_delay_2d = gt_delay[:, :, slice_idx]
        metrics["delay_pcc"] = pcc_masked(delay_map_2d, gt_delay_2d, full_mask2d)
        metrics["delay_ssim"] = masked_ssim(delay_map_2d, gt_delay_2d, full_mask2d)
        delay_pre = pre_map(delay_map_2d, gt_delay_2d)
        delay_pre[~full_mask2d] = 0.0
        delay_pre_3d = np.zeros((X, Y, Z), dtype=np.float32)
        delay_pre_3d[:, :, slice_idx] = delay_pre
        save_nifti_like(delay_pre_3d, bold_img, args.outdir / "pinn_delay_pre.nii.gz")

    if args.run_nlls:
        print("Running voxelwise NLLS baseline on training voxels using dataset 0 only...")
        yobs_first = data["yobs_vox_t"][0]
        cvr_nlls_flat, delay_nlls_flat = fit_nlls_voxelwise(
            yobs_vox_t=yobs_first,
            etco2_tr=etco2_tr,
            tr=args.tr,
            cvr_bounds=(args.cvr_min, args.cvr_max),
            delay_bounds=(args.delay_min, args.delay_max),
            ode_T=args.ode_T,
            lambda_delay_reg=1e-3,
            delay_init=0.5 * (args.delay_min + args.delay_max),
        )
        train_mask2d = data["train_mask2d"].astype(bool)
        cvr_nlls_2d = np.zeros_like(cvr_map_2d, dtype=np.float32)
        delay_nlls_2d = np.zeros_like(delay_map_2d, dtype=np.float32)
        cvr_nlls_2d[train_mask2d] = cvr_nlls_flat
        delay_nlls_2d[train_mask2d] = delay_nlls_flat

        cvr_nlls_3d = np.zeros((X, Y, Z), dtype=np.float32)
        delay_nlls_3d = np.zeros((X, Y, Z), dtype=np.float32)
        cvr_nlls_3d[:, :, slice_idx] = cvr_nlls_2d
        delay_nlls_3d[:, :, slice_idx] = delay_nlls_2d
        save_nifti_like(cvr_nlls_3d, bold_img, args.outdir / "nlls_cvr_mag.nii.gz")
        save_nifti_like(delay_nlls_3d, bold_img, args.outdir / "nlls_delay.nii.gz")

    with open(args.outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Done.")
    if metrics:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
