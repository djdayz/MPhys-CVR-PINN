
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.interpolate import interp1d
    from scipy.ndimage import gaussian_filter1d
except ImportError as e:
    raise SystemExit("Need scipy: pip install scipy") from e


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


def central_slice_index(shape_3d):
    return int(shape_3d[2] // 2)


def normalize_to_m11(x, eps=1e-8):
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax - xmin < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (2.0 * (x - xmin) / (xmax - xmin) - 1.0).astype(np.float32)


def compute_slice_percent_bold_change(bold_slice_3d, baseline_vols):

    if bold_slice_3d.ndim != 3:
        raise ValueError("bold_slice_3d must have shape (X, Y, T)")
    T = bold_slice_3d.shape[-1]
    if baseline_vols < 1 or baseline_vols > T:
        raise ValueError("baseline_vols must be in [1, T]")

    s0 = np.mean(bold_slice_3d[..., :baseline_vols], axis=-1)
    s0_safe = np.where(np.abs(s0) < 1e-6, 1e-6, s0)
    psc = 100.0 * (bold_slice_3d - s0_safe[..., None]) / s0_safe[..., None]
    return psc.astype(np.float32), s0.astype(np.float32)


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


def build_slice_mask(slice_2d_t, mask_2d=None, std_thr=1e-6):
    if mask_2d is not None:
        return (mask_2d > 0).astype(bool)
    return (np.std(slice_2d_t, axis=-1) > std_thr).astype(bool)


def collect_paths(glob_pattern):
    root = Path(glob_pattern).expanduser()
    if root.exists() and root.is_dir():
        nii_paths = sorted(root.glob("*.nii"))
        nii_gz_paths = sorted(root.glob("*.nii.gz"))
        return nii_paths + nii_gz_paths

    pattern = str(root)
    import glob as _glob

    if root.is_absolute():
        return [Path(p) for p in sorted(_glob.glob(pattern))]
    paths = sorted(Path().glob(pattern))
    if len(paths) == 0:
        paths = [Path(p) for p in sorted(_glob.glob(pattern))]
    return paths


def split_bold_paths_exact(bold_paths, n_train, n_test, seed):
    n_total = len(bold_paths)
    n_needed = n_train + n_test
    if n_total < n_needed:
        raise ValueError(
            f"Need at least {n_needed} BOLD images "
            f"({n_train} train + {n_test} test), but found {n_total}"
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)

    train_idx = perm[:n_train]
    test_idx = perm[n_train:n_train + n_test]

    train_bold = [bold_paths[i] for i in train_idx]
    test_bold = [bold_paths[i] for i in test_idx]
    return train_bold, test_bold, train_idx, test_idx


def load_split_from_json(bold_paths, split_json):
    with open(split_json, "r") as f:
        split_info = json.load(f)

    n_bold = len(bold_paths)

    if "train_indices" in split_info:
        train_indices = split_info["train_indices"]
        if not isinstance(train_indices, list) or len(train_indices) == 0:
            raise ValueError("split JSON 'train_indices' must be a non-empty list")
        if any((not isinstance(i, int)) for i in train_indices):
            raise ValueError("All entries in 'train_indices' must be integers")

        bad_idx = [i for i in train_indices if i < 0 or i >= n_bold]
        if bad_idx:
            raise ValueError(
                f"Some train_indices are out of range for {n_bold} resolved BOLD files: {bad_idx}"
            )

        train_idx = np.asarray(train_indices, dtype=np.int32)
        train_set = set(train_indices)

        if "unused_bold_paths" in split_info:
            unused_names = {Path(p).name for p in split_info["unused_bold_paths"]}
            test_idx_list = [i for i, p in enumerate(bold_paths) if i not in train_set and p.name in unused_names]
        else:
            test_idx_list = [i for i in range(n_bold) if i not in train_set]

        test_idx = np.asarray(test_idx_list, dtype=np.int32)
        train_bold = [bold_paths[i] for i in train_idx.tolist()]
        test_bold = [bold_paths[i] for i in test_idx.tolist()]
        return train_bold, test_bold, train_idx, test_idx

    if "train_bold_paths" in split_info or "train_bold" in split_info:
        key = "train_bold_paths" if "train_bold_paths" in split_info else "train_bold"
        train_names = {Path(p).name for p in split_info[key]}
        train_idx_list = [i for i, p in enumerate(bold_paths) if p.name in train_names]
        if len(train_idx_list) == 0:
            raise ValueError(
                f"split JSON has '{key}' but none matched resolved files. "
                "Check that you are pointing at the same BOLD dataset."
            )

        if "unused_bold_paths" in split_info:
            test_names = {Path(p).name for p in split_info["unused_bold_paths"]}
            test_idx_list = [i for i, p in enumerate(bold_paths) if p.name in test_names]
        elif "test_bold" in split_info:
            test_names = {Path(p).name for p in split_info["test_bold"]}
            test_idx_list = [i for i, p in enumerate(bold_paths) if p.name in test_names]
        else:
            train_set = set(train_idx_list)
            test_idx_list = [i for i in range(n_bold) if i not in train_set]

        train_idx = np.asarray(train_idx_list, dtype=np.int32)
        test_idx = np.asarray(test_idx_list, dtype=np.int32)
        train_bold = [bold_paths[i] for i in train_idx.tolist()]
        test_bold = [bold_paths[i] for i in test_idx.tolist()]
        return train_bold, test_bold, train_idx, test_idx

    raise ValueError(
        "Invalid split JSON format: need 'train_indices', 'train_bold_paths', or 'train_bold'."
    )


def pcc_masked(a, b, mask):
    aa = a[mask].astype(np.float64)
    bb = b[mask].astype(np.float64)
    if aa.size < 2:
        return float("nan")
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def mae_masked(a, b, mask):
    aa = a[mask].astype(np.float64)
    bb = b[mask].astype(np.float64)
    if aa.size == 0:
        return float("nan")
    return float(np.mean(np.abs(aa - bb)))


def rmse_masked(a, b, mask):
    aa = a[mask].astype(np.float64)
    bb = b[mask].astype(np.float64)
    if aa.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((aa - bb) ** 2)))


class VoxelPINN(nn.Module):

    def __init__(self, n_time, use_coords, hidden_dim, n_hidden_layers, cvr_bounds, delay_bounds):
        super().__init__()
        self.use_coords = use_coords
        self.n_time = int(n_time)
        self.hidden_dim = int(hidden_dim)
        self.n_hidden_layers = int(n_hidden_layers)
        self.cvr_bounds = tuple(float(v) for v in cvr_bounds)
        self.delay_bounds = tuple(float(v) for v in delay_bounds)

        in_dim = n_time + (2 if use_coords else 0)

        layers: List[nn.Module] = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

        self.cvr_min, self.cvr_max = cvr_bounds
        self.delay_min, self.delay_max = delay_bounds

    def forward(self, signal_bt, xy_b2=None):
        if self.use_coords:
            if xy_b2 is None:
                raise ValueError("xy_b2 is required when use_coords=True")
            x = torch.cat([signal_bt, xy_b2], dim=1)
        else:
            x = signal_bt

        raw = self.net(x)
        raw_cvr = raw[:, 0:1]
        raw_delay = raw[:, 1:2]

        cvr = self.cvr_min + (self.cvr_max - self.cvr_min) * torch.sigmoid(raw_cvr)
        delay = self.delay_min + (self.delay_max - self.delay_min) * torch.sigmoid(raw_delay)
        return cvr, delay


def save_checkpoint(model, out_path, args, slice_idx, train_indices, test_indices, train_paths, test_paths):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "n_time": model.n_time,
            "use_coords": model.use_coords,
            "hidden_dim": model.hidden_dim,
            "n_hidden_layers": model.n_hidden_layers,
            "cvr_bounds": model.cvr_bounds,
            "delay_bounds": model.delay_bounds,
        },
        "run_config": dict(vars(args)),
        "slice_idx": int(slice_idx),
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "train_paths": [str(p) for p in train_paths],
        "test_paths": [str(p) for p in test_paths],
    }
    torch.save(ckpt, out_path)


def load_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["model_config"]

    model = VoxelPINN(
        n_time=int(cfg["n_time"]),
        use_coords=bool(cfg["use_coords"]),
        hidden_dim=int(cfg["hidden_dim"]),
        n_hidden_layers=int(cfg["n_hidden_layers"]),
        cvr_bounds=tuple(cfg["cvr_bounds"]),
        delay_bounds=tuple(cfg["delay_bounds"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


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


def compute_drive_and_rhs(y_bt, cvr_b1, delay_b1, t_sec_1T, etco2_interp, etco2_baseline, ode_T):

    query = t_sec_1T - delay_b1
    et_shift_bt = etco2_interp(query)
    drive_bt = cvr_b1 * (et_shift_bt - float(etco2_baseline))
    rhs_bt = (drive_bt - y_bt) / float(ode_T)
    return rhs_bt, drive_bt


def euler_integrate_from_drive(drive_bt, y0_b1, ode_T, tr):

    B, Tn = drive_bt.shape
    y = torch.zeros((B, Tn), dtype=drive_bt.dtype, device=drive_bt.device)
    y[:, 0:1] = y0_b1

    alpha = float(tr) / float(ode_T)
    for t in range(Tn - 1):
        y[:, t + 1] = y[:, t] + alpha * (drive_bt[:, t] - y[:, t])
    return y


def prepare_voxel_dataset(bold_paths, gt_cvr_path, gt_delay_path, mask_3d, slice_idx, baseline_vols, tr, smooth_sigma_vols, max_voxels_per_image, seed, tag):

    if len(bold_paths) == 0:
        raise ValueError(f"No {tag} images")

    print(f"[{tag}] Inspecting first image...")
    ref_img = load_nifti(bold_paths[0])
    ref_bold = get_data(ref_img)
    if ref_bold.ndim != 4:
        raise ValueError(f"{bold_paths[0]} is not 4D")
    X, Y, Z, T = ref_bold.shape

    if not (0 <= slice_idx < Z):
        raise ValueError(f"slice_idx must be in [0, {Z-1}]")

    gt_cvr_full = get_data(load_nifti(gt_cvr_path))
    gt_delay_full = get_data(load_nifti(gt_delay_path))

    if gt_cvr_full.shape != (X, Y, Z):
        raise ValueError(f"GT CVR shape mismatch: expected {(X, Y, Z)}, got {gt_cvr_full.shape}")
    if gt_delay_full.shape != (X, Y, Z):
        raise ValueError(f"GT delay shape mismatch: expected {(X, Y, Z)}, got {gt_delay_full.shape}")

    yy, xx = np.meshgrid(np.arange(Y), np.arange(X))
    x_norm_img = normalize_to_m11(xx.astype(np.float32))
    y_norm_img = normalize_to_m11(yy.astype(np.float32))

    signal_list = []
    signal_smooth_list = []
    dy_dt_list = []
    x_list = []
    y_list = []
    cvr_gt_list = []
    delay_gt_list = []
    img_id_list = []

    rng = np.random.default_rng(seed)

    print(f"[{tag}] Preparing {len(bold_paths)} images...")
    for i, bp in enumerate(bold_paths):
        if i == 0 or (i + 1) % 10 == 0 or i == len(bold_paths) - 1:
            print(f"[{tag}] Image {i+1}/{len(bold_paths)}: {bp.name}")

        bold = get_data(load_nifti(bp))
        if bold.shape != (X, Y, Z, T):
            raise ValueError(f"BOLD shape mismatch for {bp}")

        sl_raw = bold[:, :, slice_idx, :]
        sl_psc, _ = compute_slice_percent_bold_change(sl_raw, baseline_vols=baseline_vols)

        mask2d = None if mask_3d is None else mask_3d[:, :, slice_idx]
        valid_mask = build_slice_mask(sl_psc, mask2d)

        vox_idx = np.argwhere(valid_mask)
        if vox_idx.shape[0] == 0:
            continue

        if max_voxels_per_image is not None and max_voxels_per_image < vox_idx.shape[0]:
            chosen = rng.choice(vox_idx.shape[0], size=max_voxels_per_image, replace=False)
            vox_idx = vox_idx[chosen]

        sl_smooth = gaussian_filter1d(
            sl_psc,
            sigma=float(smooth_sigma_vols),
            axis=-1,
            mode="nearest",
        )
        sl_dydt = np.gradient(sl_smooth, float(tr), axis=-1)

        for vx, vy in vox_idx:
            signal_list.append(sl_psc[vx, vy, :].astype(np.float32))
            signal_smooth_list.append(sl_smooth[vx, vy, :].astype(np.float32))
            dy_dt_list.append(sl_dydt[vx, vy, :].astype(np.float32))
            x_list.append(float(x_norm_img[vx, vy]))
            y_list.append(float(y_norm_img[vx, vy]))
            cvr_gt_list.append(float(gt_cvr_full[vx, vy, slice_idx]))
            delay_gt_list.append(float(gt_delay_full[vx, vy, slice_idx]))
            img_id_list.append(i)

    if len(signal_list) == 0:
        raise RuntimeError(f"[{tag}] No voxel samples collected.")

    t_sec = np.arange(T, dtype=np.float32) * float(tr)

    out = {
        "signal": np.stack(signal_list, axis=0).astype(np.float32),
        "signal_smooth": np.stack(signal_smooth_list, axis=0).astype(np.float32),
        "dy_dt": np.stack(dy_dt_list, axis=0).astype(np.float32),
        "x": np.asarray(x_list, dtype=np.float32),
        "y": np.asarray(y_list, dtype=np.float32),
        "gt_cvr": np.asarray(cvr_gt_list, dtype=np.float32),
        "gt_delay": np.asarray(delay_gt_list, dtype=np.float32),
        "img_id": np.asarray(img_id_list, dtype=np.int32),
        "t_sec": t_sec.astype(np.float32),
        "shape_4d": np.array([X, Y, Z, T], dtype=np.int32),
        "slice_idx": np.array(slice_idx, dtype=np.int32),
        "ref_bold_path": np.array(str(bold_paths[0]), dtype=object),
    }

    print(f"[{tag}] Done. Samples: {out['signal'].shape[0]}")
    return out


@torch.no_grad()
def evaluate_dataset(model, data, etco2_interp, etco2_baseline, ode_T, tr, use_coords, lambda_param_cvr, lambda_param_delay, lambda_data, lambda_phys, device, batch_size):
    model.eval()

    signal = torch.tensor(data["signal"], dtype=torch.float32, device=device)
    signal_smooth = torch.tensor(data["signal_smooth"], dtype=torch.float32, device=device)
    dy_dt = torch.tensor(data["dy_dt"], dtype=torch.float32, device=device)
    gt_cvr = torch.tensor(data["gt_cvr"], dtype=torch.float32, device=device).unsqueeze(1)
    gt_delay = torch.tensor(data["gt_delay"], dtype=torch.float32, device=device).unsqueeze(1)
    xy = torch.stack([
        torch.tensor(data["x"], dtype=torch.float32, device=device),
        torch.tensor(data["y"], dtype=torch.float32, device=device),
    ], dim=1)
    t_sec = torch.tensor(data["t_sec"], dtype=torch.float32, device=device).unsqueeze(0)

    N = signal.shape[0]
    sums = {
        "total": 0.0,
        "param_cvr": 0.0,
        "param_delay": 0.0,
        "data": 0.0,
        "phys": 0.0,
    }
    n_seen = 0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start

        sb = signal[start:end]
        sb_smooth = signal_smooth[start:end]
        dydt_b = dy_dt[start:end]
        cvr_gt_b = gt_cvr[start:end]
        delay_gt_b = gt_delay[start:end]
        xy_b = xy[start:end]

        cvr_hat, delay_hat = model(sb, xy_b if use_coords else None)

        rhs_b, drive_b = compute_drive_and_rhs(
            y_bt=sb_smooth,
            cvr_b1=cvr_hat,
            delay_b1=delay_hat,
            t_sec_1T=t_sec,
            etco2_interp=etco2_interp,
            etco2_baseline=etco2_baseline,
            ode_T=ode_T,
        )

        y_pred = euler_integrate_from_drive(
            drive_bt=drive_b,
            y0_b1=sb[:, 0:1],
            ode_T=ode_T,
            tr=tr,
        )

        loss_param_cvr = F.mse_loss(cvr_hat, cvr_gt_b)
        loss_param_delay = F.mse_loss(delay_hat, delay_gt_b)
        loss_data = F.mse_loss(y_pred, sb)
        loss_phys = F.mse_loss(rhs_b, dydt_b)

        loss_total = (
            float(lambda_param_cvr) * loss_param_cvr
            + float(lambda_param_delay) * loss_param_delay
            + float(lambda_data) * loss_data
            + float(lambda_phys) * loss_phys
        )

        sums["total"] += float(loss_total.item()) * B
        sums["param_cvr"] += float(loss_param_cvr.item()) * B
        sums["param_delay"] += float(loss_param_delay.item()) * B
        sums["data"] += float(loss_data.item()) * B
        sums["phys"] += float(loss_phys.item()) * B
        n_seen += B

    return {k: v / max(n_seen, 1) for k, v in sums.items()}


def train_model(train_data, test_data, etco2_tr, tr, use_coords, hidden_dim, n_hidden_layers, cvr_bounds, delay_bounds, ode_T, epochs, batch_size, lr, weight_decay, lambda_param_cvr, lambda_param_delay, lambda_data, lambda_phys, device, log_every):
    n_time = int(train_data["signal"].shape[1])

    model = VoxelPINN(
        n_time=n_time,
        use_coords=use_coords,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        cvr_bounds=cvr_bounds,
        delay_bounds=delay_bounds,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    signal = torch.tensor(train_data["signal"], dtype=torch.float32, device=device)
    signal_smooth = torch.tensor(train_data["signal_smooth"], dtype=torch.float32, device=device)
    dy_dt = torch.tensor(train_data["dy_dt"], dtype=torch.float32, device=device)
    gt_cvr = torch.tensor(train_data["gt_cvr"], dtype=torch.float32, device=device).unsqueeze(1)
    gt_delay = torch.tensor(train_data["gt_delay"], dtype=torch.float32, device=device).unsqueeze(1)
    xy = torch.stack([
        torch.tensor(train_data["x"], dtype=torch.float32, device=device),
        torch.tensor(train_data["y"], dtype=torch.float32, device=device),
    ], dim=1)
    t_sec_np = train_data["t_sec"].astype(np.float32)
    t_sec = torch.tensor(t_sec_np, dtype=torch.float32, device=device).unsqueeze(0)

    etco2_baseline = float(np.mean(etco2_tr[:max(1, min(30, len(etco2_tr)))]))
    etco2_interp = TorchLinearInterpolator1D(t_sec_np, etco2_tr.astype(np.float32)).to(device)

    N = signal.shape[0]
    history: List[Dict[str, float]] = []

    best_test_loss = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()

        perm = torch.randperm(N, device=device)

        sums = {
            "total": 0.0,
            "param_cvr": 0.0,
            "param_delay": 0.0,
            "data": 0.0,
            "phys": 0.0,
        }
        n_seen = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = perm[start:end]
            B = idx.numel()

            sb = signal[idx]
            sb_smooth = signal_smooth[idx]
            dydt_b = dy_dt[idx]
            cvr_gt_b = gt_cvr[idx]
            delay_gt_b = gt_delay[idx]
            xy_b = xy[idx]

            cvr_hat, delay_hat = model(sb, xy_b if use_coords else None)

            rhs_b, drive_b = compute_drive_and_rhs(
                y_bt=sb_smooth,
                cvr_b1=cvr_hat,
                delay_b1=delay_hat,
                t_sec_1T=t_sec,
                etco2_interp=etco2_interp,
                etco2_baseline=etco2_baseline,
                ode_T=ode_T,
            )

            y_pred = euler_integrate_from_drive(
                drive_bt=drive_b,
                y0_b1=sb[:, 0:1],
                ode_T=ode_T,
                tr=tr,
            )

            loss_param_cvr = F.mse_loss(cvr_hat, cvr_gt_b)
            loss_param_delay = F.mse_loss(delay_hat, delay_gt_b)
            loss_data = F.mse_loss(y_pred, sb)
            loss_phys = F.mse_loss(rhs_b, dydt_b)

            loss_total = (
                float(lambda_param_cvr) * loss_param_cvr
                + float(lambda_param_delay) * loss_param_delay
                + float(lambda_data) * loss_data
                + float(lambda_phys) * loss_phys
            )

            opt.zero_grad(set_to_none=True)
            loss_total.backward()
            opt.step()

            sums["total"] += float(loss_total.item()) * B
            sums["param_cvr"] += float(loss_param_cvr.item()) * B
            sums["param_delay"] += float(loss_param_delay.item()) * B
            sums["data"] += float(loss_data.item()) * B
            sums["phys"] += float(loss_phys.item()) * B
            n_seen += B

        train_metrics = {k: v / max(n_seen, 1) for k, v in sums.items()}

        test_metrics = evaluate_dataset(
            model=model,
            data=test_data,
            etco2_interp=etco2_interp,
            etco2_baseline=etco2_baseline,
            ode_T=ode_T,
            tr=tr,
            use_coords=use_coords,
            lambda_param_cvr=lambda_param_cvr,
            lambda_param_delay=lambda_param_delay,
            lambda_data=lambda_data,
            lambda_phys=lambda_phys,
            device=device,
            batch_size=batch_size,
        )

        row = {
            "epoch": float(epoch),
            "train_total_loss": float(train_metrics["total"]),
            "train_param_cvr_loss": float(train_metrics["param_cvr"]),
            "train_param_delay_loss": float(train_metrics["param_delay"]),
            "train_data_loss": float(train_metrics["data"]),
            "train_phys_loss": float(train_metrics["phys"]),
            "test_total_loss": float(test_metrics["total"]),
            "test_param_cvr_loss": float(test_metrics["param_cvr"]),
            "test_param_delay_loss": float(test_metrics["param_delay"]),
            "test_data_loss": float(test_metrics["data"]),
            "test_phys_loss": float(test_metrics["phys"]),
            "epoch_time_sec": float(time.time() - t0),
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)

        current_test_loss = row["test_total_loss"]
        if current_test_loss < best_test_loss:
            best_test_loss = current_test_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[best] epoch={epoch} test_total={best_test_loss:.6e}")

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"[Epoch {epoch:4d}/{epochs}] "
                f"train_total={row['train_total_loss']:.6e} "
                f"train_cvr={row['train_param_cvr_loss']:.6e} "
                f"train_delay={row['train_param_delay_loss']:.6e} "
                f"train_data={row['train_data_loss']:.6e} "
                f"train_phys={row['train_phys_loss']:.6e} "
                f"test_total={row['test_total_loss']:.6e} "
                f"time={row['epoch_time_sec']:.2f}s"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model from epoch {best_epoch} with test_total={best_test_loss:.6e}")

    return model, history


@torch.no_grad()
def infer_single_image_slice(model, bold_path, mask_3d, slice_idx, baseline_vols, use_coords, device, batch_size=2048):
    ref_img = load_nifti(bold_path)
    bold = get_data(ref_img)
    if bold.ndim != 4:
        raise ValueError(f"{bold_path} is not 4D")

    X, Y, Z, T = bold.shape
    if not (0 <= slice_idx < Z):
        raise ValueError(f"Invalid slice_idx {slice_idx}")

    sl_raw = bold[:, :, slice_idx, :]
    sl_psc, _ = compute_slice_percent_bold_change(sl_raw, baseline_vols=baseline_vols)

    mask2d = None if mask_3d is None else mask_3d[:, :, slice_idx]
    valid_mask = build_slice_mask(sl_psc, mask2d)

    yy, xx = np.meshgrid(np.arange(Y), np.arange(X))
    x_norm_img = normalize_to_m11(xx.astype(np.float32))
    y_norm_img = normalize_to_m11(yy.astype(np.float32))

    vox_idx = np.argwhere(valid_mask)
    if vox_idx.shape[0] == 0:
        raise RuntimeError("No valid voxels on slice")

    signals = np.stack([sl_psc[vx, vy, :] for vx, vy in vox_idx], axis=0).astype(np.float32)
    xy = np.stack([[x_norm_img[vx, vy], y_norm_img[vx, vy]] for vx, vy in vox_idx], axis=0).astype(np.float32)

    signals_t = torch.tensor(signals, dtype=torch.float32, device=device)
    xy_t = torch.tensor(xy, dtype=torch.float32, device=device)

    cvr_out = np.zeros((vox_idx.shape[0],), dtype=np.float32)
    delay_out = np.zeros((vox_idx.shape[0],), dtype=np.float32)

    for start in range(0, vox_idx.shape[0], batch_size):
        end = min(start + batch_size, vox_idx.shape[0])
        cvr_b, delay_b = model(signals_t[start:end], xy_t[start:end] if use_coords else None)
        cvr_out[start:end] = cvr_b[:, 0].cpu().numpy()
        delay_out[start:end] = delay_b[:, 0].cpu().numpy()

    cvr_map_2d = np.zeros((X, Y), dtype=np.float32)
    delay_map_2d = np.zeros((X, Y), dtype=np.float32)
    for i, (vx, vy) in enumerate(vox_idx):
        cvr_map_2d[vx, vy] = cvr_out[i]
        delay_map_2d[vx, vy] = delay_out[i]

    return cvr_map_2d, delay_map_2d, ref_img


@torch.no_grad()
def evaluate_unseen_images(model, bold_paths, gt_cvr_path, gt_delay_path, mask_3d, slice_idx, baseline_vols, use_coords, device, batch_size, out_csv):
    gt_cvr_full = get_data(load_nifti(gt_cvr_path))
    gt_delay_full = get_data(load_nifti(gt_delay_path))
    gt_cvr = gt_cvr_full[:, :, slice_idx]
    gt_delay = gt_delay_full[:, :, slice_idx]

    rows = []
    for i, bp in enumerate(bold_paths):
        print(f"[test-eval] {i+1}/{len(bold_paths)}: {bp.name}")

        pred_cvr_2d, pred_delay_2d, _ = infer_single_image_slice(
            model=model,
            bold_path=bp,
            mask_3d=mask_3d,
            slice_idx=slice_idx,
            baseline_vols=baseline_vols,
            use_coords=use_coords,
            device=device,
            batch_size=batch_size,
        )

        if mask_3d is not None:
            mask2d = mask_3d[:, :, slice_idx] > 0
        else:
            mask2d = np.isfinite(gt_cvr) & np.isfinite(gt_delay)

        row = {
            "image": bp.name,
            "cvr_mae": mae_masked(pred_cvr_2d, gt_cvr, mask2d),
            "cvr_rmse": rmse_masked(pred_cvr_2d, gt_cvr, mask2d),
            "cvr_pcc": pcc_masked(pred_cvr_2d, gt_cvr, mask2d),
            "delay_mae": mae_masked(pred_delay_2d, gt_delay, mask2d),
            "delay_rmse": rmse_masked(pred_delay_2d, gt_delay, mask2d),
            "delay_pcc": pcc_masked(pred_delay_2d, gt_delay, mask2d),
        }
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

@torch.no_grad()
def save_unseen_image_maps(model, bold_paths, mask_3d, slice_idx, baseline_vols, use_coords, device, batch_size, out_dir):

    out_dir.mkdir(parents=True, exist_ok=True)

    for i, bp in enumerate(bold_paths):
        print(f"[test-maps] {i+1}/{len(bold_paths)}: {bp.name}")

        cvr_map_2d, delay_map_2d, ref_img = infer_single_image_slice(
            model=model,
            bold_path=bp,
            mask_3d=mask_3d,
            slice_idx=slice_idx,
            baseline_vols=baseline_vols,
            use_coords=use_coords,
            device=device,
            batch_size=batch_size,
        )

        bold = get_data(ref_img)
        if bold.ndim != 4:
            raise ValueError(f"{bp} is not 4D")

        X, Y, Z, _ = bold.shape

        cvr_3d = np.zeros((X, Y, Z), dtype=np.float32)
        delay_3d = np.zeros((X, Y, Z), dtype=np.float32)

        cvr_3d[:, :, slice_idx] = cvr_map_2d
        delay_3d[:, :, slice_idx] = delay_map_2d

        stem = bp.name.replace(".nii.gz", "").replace(".nii", "")

        save_nifti_like(
            cvr_3d,
            ref_img,
            out_dir / f"{stem}_pinn_cvr_mag.nii.gz",
        )

        save_nifti_like(
            delay_3d,
            ref_img,
            out_dir / f"{stem}_pinn_delay.nii.gz",
        )


def save_history_csv(history, out_csv):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if len(history) == 0:
        return
    keys = list(history[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_history_txt(history, out_txt):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        f.write(
            "# epoch train_total train_cvr train_delay train_data train_phys "
            "test_total test_cvr test_delay test_data test_phys epoch_time_sec lr\n"
        )
        for row in history:
            f.write(
                f"{int(row['epoch'])} "
                f"{row['train_total_loss']:.8e} "
                f"{row['train_param_cvr_loss']:.8e} "
                f"{row['train_param_delay_loss']:.8e} "
                f"{row['train_data_loss']:.8e} "
                f"{row['train_phys_loss']:.8e} "
                f"{row['test_total_loss']:.8e} "
                f"{row['test_param_cvr_loss']:.8e} "
                f"{row['test_param_delay_loss']:.8e} "
                f"{row['test_data_loss']:.8e} "
                f"{row['test_phys_loss']:.8e} "
                f"{row['epoch_time_sec']:.6f} "
                f"{row['lr']:.8e}\n"
            )


def main():
    p = argparse.ArgumentParser(
        description=(
            "Train on 60 random simulated BOLD images sharing one GT CVR map and one GT delay map, "
            "save best checkpoint, and apply to 40 unseen images or real data."
        )
    )

    p.add_argument("--train-bold-glob", type=str, default="/workspace/tCNR_0.500",
                   help="Directory or glob for all simulated BOLD images")
    p.add_argument("--gt-cvr", type=Path, default=Path("/workspace/gt_cvr_mag_2p5mm.nii"),
                   help="Single ground-truth CVR map used for all simulated BOLD images")
    p.add_argument("--gt-delay", type=Path, default=Path("/workspace/gt_delay_2p5mm.nii"),
                   help="Single ground-truth delay map used for all simulated BOLD images")

    p.add_argument("--infer-bold-glob", type=str, default=None,
                   help="Optional unseen simulated or real BOLD directory/glob to infer after training")
    p.add_argument("--split-json", type=Path, default=Path("/workspace/image_split.json"),
                   help="Optional data_split.json or similar split file to reuse the same train/test BOLD selection")

    p.add_argument("--outdir", type=Path, default=Path("/workspace/sup_0.5"))

    p.add_argument("--mask", type=Path, default=None)
    p.add_argument("--slice-index", type=int, default=62)
    p.add_argument("--baseline-vols", type=int, default=30)
    p.add_argument("--tr", type=float, default=1.55)

    p.add_argument("--extra-pre", type=int, default=31)
    p.add_argument("--extra-post", type=int, default=93)
    p.add_argument("--etco2-sample-dt", type=float, default=1.0)

    p.add_argument("--smooth-sigma-vols", type=float, default=1.0)
    p.add_argument("--max-voxels-per-image", type=int, default=2900)

    p.add_argument("--n-train-images", type=int, default=60)
    p.add_argument("--n-test-images", type=int, default=40)

    p.add_argument("--use-coords", action="store_true")

    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--n-hidden-layers", type=int, default=2)

    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)

    p.add_argument("--lambda-param-cvr", type=float, default=1.0)
    p.add_argument("--lambda-param-delay", type=float, default=0.01)
    p.add_argument("--lambda-data", type=float, default=1.0)
    p.add_argument("--lambda-phys", type=float, default=0.5)

    p.add_argument("--ode-T", type=float, default=15.0)
    p.add_argument("--cvr-min", type=float, default=-0.4)
    p.add_argument("--cvr-max", type=float, default=1.8)
    p.add_argument("--delay-min", type=float, default=0.0)
    p.add_argument("--delay-max", type=float, default=100.0)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")

    args = p.parse_args()
    set_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

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

    bold_paths = collect_paths(args.train_bold_glob)
    if len(bold_paths) == 0:
        raise RuntimeError("No BOLD images found.")

    gt_cvr_path = args.gt_cvr
    gt_delay_path = args.gt_delay

    if not gt_cvr_path.exists():
        raise RuntimeError(f"GT CVR file not found: {gt_cvr_path}")
    if not gt_delay_path.exists():
        raise RuntimeError(f"GT delay file not found: {gt_delay_path}")

    print(f"Found {len(bold_paths)} simulated BOLD images")
    print(f"Using shared GT CVR:   {gt_cvr_path}")
    print(f"Using shared GT delay: {gt_delay_path}")

    if args.split_json is not None:
        train_img_paths, test_img_paths, train_indices, test_indices = load_split_from_json(
            bold_paths=bold_paths,
            split_json=args.split_json,
        )
        print(f"Loaded split from: {args.split_json}")
    else:
        train_img_paths, test_img_paths, train_indices, test_indices = split_bold_paths_exact(
            bold_paths=bold_paths,
            n_train=args.n_train_images,
            n_test=args.n_test_images,
            seed=args.seed,
        )

    print(f"Train images: {len(train_img_paths)}")
    print(f"Test images:  {len(test_img_paths)}")

    with open(args.outdir / "image_split.json", "w") as f:
        json.dump(
            {
                "train_indices": train_indices.tolist(),
                "test_indices": test_indices.tolist(),
                "train_bold": [str(p) for p in train_img_paths],
                "test_bold": [str(p) for p in test_img_paths],
                "gt_cvr": str(gt_cvr_path),
                "gt_delay": str(gt_delay_path),
                "source_split_json": str(args.split_json) if args.split_json is not None else None,
            },
            f,
            indent=2,
        )

    ref_img = load_nifti(train_img_paths[0])
    ref_bold = get_data(ref_img)
    if ref_bold.ndim != 4:
        raise ValueError("Training BOLD must be 4D")
    X, Y, Z, T = ref_bold.shape

    slice_idx = central_slice_index((X, Y, Z)) if args.slice_index < 0 else args.slice_index
    if not (0 <= slice_idx < Z):
        raise ValueError(f"Invalid slice index {slice_idx}")

    mask = None
    if args.mask is not None:
        mask = get_data(load_nifti(args.mask))
        if mask.shape != (X, Y, Z):
            raise ValueError(f"Mask shape {mask.shape} does not match BOLD shape {(X, Y, Z)}")

    etco2_raw = build_long_etco2(extra_pre=args.extra_pre, extra_post=args.extra_post)
    etco2_tr = resample_signal_to_tr(
        signal=etco2_raw,
        signal_dt=args.etco2_sample_dt,
        n_time=T,
        tr=args.tr,
    )

    print(f"Slice index: {slice_idx}")
    print(f"EtCO2 raw samples: {len(etco2_raw)}")
    print(f"EtCO2 TR samples: {len(etco2_tr)}")

    train_data = prepare_voxel_dataset(
        bold_paths=train_img_paths,
        gt_cvr_path=gt_cvr_path,
        gt_delay_path=gt_delay_path,
        mask_3d=mask,
        slice_idx=slice_idx,
        baseline_vols=args.baseline_vols,
        tr=args.tr,
        smooth_sigma_vols=args.smooth_sigma_vols,
        max_voxels_per_image=args.max_voxels_per_image,
        seed=args.seed,
        tag="train",
    )

    test_data = prepare_voxel_dataset(
        bold_paths=test_img_paths,
        gt_cvr_path=gt_cvr_path,
        gt_delay_path=gt_delay_path,
        mask_3d=mask,
        slice_idx=slice_idx,
        baseline_vols=args.baseline_vols,
        tr=args.tr,
        smooth_sigma_vols=args.smooth_sigma_vols,
        max_voxels_per_image=args.max_voxels_per_image,
        seed=args.seed + 1,
        tag="test",
    )

    print(f"Train voxel samples: {train_data['signal'].shape[0]}")
    print(f"Test voxel samples:  {test_data['signal'].shape[0]}")
    print(f"Time points:         {train_data['signal'].shape[1]}")

    model, history = train_model(
        train_data=train_data,
        test_data=test_data,
        etco2_tr=etco2_tr,
        tr=args.tr,
        use_coords=args.use_coords,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        cvr_bounds=(args.cvr_min, args.cvr_max),
        delay_bounds=(args.delay_min, args.delay_max),
        ode_T=args.ode_T,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_param_cvr=args.lambda_param_cvr,
        lambda_param_delay=args.lambda_param_delay,
        lambda_data=args.lambda_data,
        lambda_phys=args.lambda_phys,
        device=device,
        log_every=max(1, args.epochs // 20),
    )

    save_checkpoint(
        model=model,
        out_path=args.outdir / "voxel_pinn_checkpoint.pt",
        args=args,
        slice_idx=slice_idx,
        train_indices=train_indices,
        test_indices=test_indices,
        train_paths=train_img_paths,
        test_paths=test_img_paths,
    )

    torch.save(model.state_dict(), args.outdir / "best_voxel_pinn_model_state_dict.pt")

    with open(args.outdir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    save_history_csv(history, args.outdir / "train_history.csv")
    save_history_txt(history, args.outdir / "train_history.txt")

    evaluate_unseen_images(
        model=model,
        bold_paths=test_img_paths,
        gt_cvr_path=gt_cvr_path,
        gt_delay_path=gt_delay_path,
        mask_3d=mask,
        slice_idx=slice_idx,
        baseline_vols=args.baseline_vols,
        use_coords=args.use_coords,
        device=device,
        batch_size=max(1024, args.batch_size),
        out_csv=args.outdir / "test_image_metrics.csv",
    )


    save_unseen_image_maps(
        model=model,
        bold_paths=test_img_paths,
        mask_3d=mask,
        slice_idx=slice_idx,
        baseline_vols=args.baseline_vols,
        use_coords=args.use_coords,
        device=device,
        batch_size=max(1024, args.batch_size),
        out_dir=args.outdir / "test_inference_maps",
    )

    if args.infer_bold_glob is not None:
        infer_paths = collect_paths(args.infer_bold_glob)
        infer_dir = args.outdir / "inference_maps"
        infer_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running inference on {len(infer_paths)} external images...")
        for i, bp in enumerate(infer_paths):
            print(f"[infer] {i+1}/{len(infer_paths)}: {bp.name}")
            cvr_map_2d, delay_map_2d, infer_ref = infer_single_image_slice(
                model=model,
                bold_path=bp,
                mask_3d=mask,
                slice_idx=slice_idx,
                baseline_vols=args.baseline_vols,
                use_coords=args.use_coords,
                device=device,
                batch_size=max(1024, args.batch_size),
            )

            infer_bold = get_data(infer_ref)
            if infer_bold.ndim != 4:
                raise ValueError(f"{bp} is not 4D")
            Xi, Yi, Zi, _ = infer_bold.shape

            cvr_3d = np.zeros((Xi, Yi, Zi), dtype=np.float32)
            delay_3d = np.zeros((Xi, Yi, Zi), dtype=np.float32)
            cvr_3d[:, :, slice_idx] = cvr_map_2d
            delay_3d[:, :, slice_idx] = delay_map_2d

            stem = bp.name.replace(".nii.gz", "").replace(".nii", "")
            save_nifti_like(cvr_3d, infer_ref, infer_dir / f"{stem}_pinn_cvr_mag.nii.gz")
            save_nifti_like(delay_3d, infer_ref, infer_dir / f"{stem}_pinn_delay.nii.gz")

    print("Done.")
    print(f"Outputs saved in: {args.outdir}")


if __name__ == "__main__":
    main()
