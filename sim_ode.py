
import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output

try:
    from scipy.ndimage import gaussian_filter, binary_closing, binary_dilation
except ImportError as e:
    raise SystemExit("Need scipy: pip install scipy") from e


ROI_KEYS = ["vessel", "vcsf", "cgm", "sgm", "wm"]


def load_nii(path):
    return nib.load(str(path))


def get_data(img):
    return np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)


def save_nii(data, ref, out_path, tr=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hdr = ref.header.copy()
    img = nib.Nifti1Image(data.astype(np.float32), ref.affine, hdr)
    img.header.set_data_shape(data.shape)

    if data.ndim == 4:
        zooms3 = hdr.get_zooms()[:3]
        if tr is None:
            tr = 1.0
        img.header.set_zooms(zooms3 + (float(tr),))

    nib.save(img, str(out_path))


def ensure_same_shape(a, b, name_a, name_b):
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {name_a} {a.shape} vs {name_b} {b.shape}")


def load_s0_json(path):
    with open(path, "r") as f:
        return json.load(f)


def choose_s0_values(meta, stat):
    stat = stat.lower().strip()
    if stat == "median":
        key = "s0_by_roi_median"
    elif stat == "mean":
        key = "s0_by_roi_mean"
    else:
        raise ValueError("s0-stat must be 'median' or 'mean'")

    if key not in meta:
        raise KeyError(f"{key} not found in JSON")

    vals = meta[key]
    out: Dict[str, float] = {}
    for roi in ROI_KEYS:
        if roi not in vals:
            raise KeyError(f"ROI '{roi}' missing from {key}")
        out[roi] = float(vals[roi])
    return out


def load_roi_masks(vessel, vcsf, cgm, sgm, wm):
    imgs = {
        "vessel": load_nii(vessel),
        "vcsf": load_nii(vcsf),
        "cgm": load_nii(cgm),
        "sgm": load_nii(sgm),
        "wm": load_nii(wm),
    }
    ref_img = imgs["cgm"]
    ref_data = get_data(ref_img)

    masks: Dict[str, np.ndarray] = {}
    for k, img in imgs.items():
        arr = get_data(img) > 0
        ensure_same_shape(ref_data, arr.astype(np.float32), "ref_roi", f"{k}_mask")
        masks[k] = arr
    return ref_img, masks


def apply_previous_vessel_fill_setting(masks, vessel_close_iter=2, brain_dilate_iter=2):

    out = {k: v.copy() for k, v in masks.items()}

    vessel = out["vessel"]
    if vessel_close_iter > 0:
        vessel = binary_closing(vessel, iterations=int(vessel_close_iter))

    roi_union = np.zeros_like(vessel, dtype=bool)
    for roi in ROI_KEYS:
        roi_union |= out[roi]

    brain_support = roi_union
    if brain_dilate_iter > 0:
        brain_support = binary_dilation(brain_support, iterations=int(brain_dilate_iter))

    missing_inside_brain = brain_support & (~roi_union)
    vessel = vessel | missing_inside_brain

    out["vessel"] = vessel
    return out


def build_exclusive_roi_masks(masks, priority):
    occupied = np.zeros_like(next(iter(masks.values())), dtype=bool)
    exclusive: Dict[str, np.ndarray] = {}

    for roi in priority:
        m = masks[roi] & (~occupied)
        exclusive[roi] = m
        occupied |= masks[roi]
    return exclusive


def build_s0_map_from_rois(exclusive_masks, s0_vals):
    shape = next(iter(exclusive_masks.values())).shape
    s0 = np.zeros(shape, dtype=np.float32)
    for roi in ROI_KEYS:
        s0[exclusive_masks[roi]] = np.float32(s0_vals[roi])
    return s0


def build_long_etco2(extra_pre, extra_post):
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


def crop_scan_etco2(long_etco2, extra_pre, extra_post, min_length=None, pad_value=40.0):
    if extra_post == 0:
        out = long_etco2[extra_pre:].astype(np.float32)
    else:
        out = long_etco2[extra_pre:-extra_post].astype(np.float32)

    if min_length is not None and out.size < int(min_length):
        pad_len = int(min_length) - out.size
        out = np.concatenate(
            [out, np.float32(pad_value) * np.ones(pad_len, dtype=np.float32)],
            axis=0,
        )

    return out.astype(np.float32)


def resample_1d(signal, t_old, t_new):
    return np.interp(t_new, t_old, signal).astype(np.float32)


def voxel_sizes_from_affine(affine):
    return tuple(np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).tolist())


def downsample_map_to_voxel_size(img, order, voxel_size_mm):
    data = get_data(img)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D image, got {data.shape}")

    tmp_img = nib.Nifti1Image(data.astype(np.float32), img.affine, img.header)
    out_img = resample_to_output(
        tmp_img,
        voxel_sizes=(voxel_size_mm, voxel_size_mm, voxel_size_mm),
        order=order,
    )
    return out_img


def smooth_4d_spatial_only(bold4d, voxel_sizes_mm, fwhm_mm, mask=None):

    if fwhm_mm <= 0:
        out = bold4d.astype(np.float32, copy=True)
        if mask is not None:
            out[~mask, :] = 0.0
        return out

    sigma_mm = float(fwhm_mm) / 2.354820045
    sigma_vox = tuple(sigma_mm / max(v, 1e-6) for v in voxel_sizes_mm)

    out = gaussian_filter(
        bold4d,
        sigma=(sigma_vox[0], sigma_vox[1], sigma_vox[2], 0.0),
        mode="nearest",
    ).astype(np.float32)

    if mask is not None:
        out[~mask, :] = 0.0

    return out


def shift_signal_by_steps(u, lag_steps):
    out = np.empty_like(u)

    if lag_steps > 0:
        out[:lag_steps] = u[0]
        out[lag_steps:] = u[:-lag_steps]
    elif lag_steps < 0:
        s = -lag_steps
        out[:-s] = u[s:]
        out[-s:] = u[-1]
    else:
        out[:] = u
    return out


def solve_unit_response(u_shifted, T_sec, internal_dt):

    nt = u_shifted.size
    q = np.zeros(nt, dtype=np.float32)
    alpha = np.float32(internal_dt / max(T_sec, 1e-6))

    for k in range(1, nt):
        q[k] = q[k - 1] + alpha * (u_shifted[k - 1] - q[k - 1])

    return q


def build_lag_response_dict(u_delta, tau_flat_sec, internal_dt, T_sec, t_internal, t_out):
    lag_steps_flat = np.rint(tau_flat_sec / float(internal_dt)).astype(np.int32)
    unique_lags = np.unique(lag_steps_flat)

    q_out_dict: Dict[int, np.ndarray] = {}
    for lag in unique_lags:
        u_shifted = shift_signal_by_steps(u_delta, int(lag))
        q_internal = solve_unit_response(u_shifted, T_sec=T_sec, internal_dt=internal_dt)
        q_out = np.interp(t_out, t_internal, q_internal).astype(np.float32)
        q_out_dict[int(lag)] = q_out

    return lag_steps_flat, q_out_dict


def construct_clean_bold_chunked(cvr_frac_flat, s0_flat, lag_steps_flat, q_out_dict, mask_flat, out_shape_3d, nt_out, chunk_vox=50000):
    shape4d = (*out_shape_3d, nt_out)
    bold4d = np.zeros(shape4d, dtype=np.float32)
    flat_view = bold4d.reshape(-1, nt_out)

    nvox = cvr_frac_flat.size
    for start in range(0, nvox, chunk_vox):
        end = min(start + chunk_vox, nvox)

        cvr_chunk = cvr_frac_flat[start:end]
        s0_chunk = s0_flat[start:end]
        lag_chunk = lag_steps_flat[start:end]
        mask_chunk = mask_flat[start:end]

        out_chunk = np.zeros((end - start, nt_out), dtype=np.float32)

        unique_lags = np.unique(lag_chunk)
        for lag in unique_lags:
            idx_local = np.where(lag_chunk == lag)[0]
            q = q_out_dict[int(lag)]
            out_chunk[idx_local, :] = s0_chunk[idx_local, None] * (
                1.0 + cvr_chunk[idx_local, None] * q[None, :]
            )

        out_chunk[~mask_chunk, :] = 0.0
        flat_view[start:end, :] = out_chunk

    return bold4d


def steady_time_masks(t_out):
    norm_mask = (t_out >= 30.0) & (t_out < 90.0)
    hyper_mask = (t_out >= 180.0) & (t_out < 240.0)
    return norm_mask, hyper_mask


def compute_noise_sd_from_tcnr(bold4d_clean, t_out, mask, tcnr):
    if tcnr <= 0:
        return np.zeros(bold4d_clean.shape[:3], dtype=np.float32)

    norm_mask, hyper_mask = steady_time_masks(t_out)
    mean_norm = bold4d_clean[..., norm_mask].mean(axis=-1)
    mean_hyper = bold4d_clean[..., hyper_mask].mean(axis=-1)
    delta = np.maximum(mean_hyper - mean_norm, 0.0)
    noise_sd = delta / float(tcnr)
    noise_sd[~mask] = 0.0
    return noise_sd.astype(np.float32)


def add_noise_repetition_chunked(bold4d_clean, noise_sd, mask, rng, chunk_vox=50000):
    nt = bold4d_clean.shape[-1]
    bold_rep = np.empty_like(bold4d_clean)

    flat_clean = bold4d_clean.reshape(-1, nt)
    flat_rep = bold_rep.reshape(-1, nt)
    flat_noise_sd = noise_sd.reshape(-1)
    flat_mask = mask.reshape(-1)

    nvox = flat_noise_sd.size
    for start in range(0, nvox, chunk_vox):
        end = min(start + chunk_vox, nvox)

        clean_chunk = flat_clean[start:end, :]
        sd_chunk = flat_noise_sd[start:end]
        mask_chunk = flat_mask[start:end]

        if np.all(sd_chunk == 0):
            rep_chunk = clean_chunk.copy()
        else:
            noise_chunk = rng.standard_normal(size=(end - start, nt)).astype(np.float32)
            noise_chunk *= sd_chunk[:, None]
            rep_chunk = clean_chunk + noise_chunk

        rep_chunk[~mask_chunk, :] = 0.0
        flat_rep[start:end, :] = rep_chunk

    return bold_rep.astype(np.float32)


def parse_tcnr_list(s):
    vals = []
    for x in s.split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    if not vals:
        raise ValueError("Empty --tcnr-list")
    return vals


def main():
    p = argparse.ArgumentParser(
        description="Memory-safe chunked dynamic BOLD simulator with fixed output timepoints"
    )

    p.add_argument(
        "--cvr-mag",
        type=Path,
        default=Path("/Users/mac/PycharmProjects/pythonMPhysproject/hv_dist/mida_maps/CVR_mag_mida.nii.gz"),
        help="High-resolution CVR magnitude map in %/mmHg",
    )
    p.add_argument(
        "--delay",
        type=Path,
        default=Path("/Users/mac/PycharmProjects/pythonMPhysproject/hv_dist/mida_maps/CVR_delay_mida.nii.gz"),
        help="High-resolution delay map in seconds",
    )

    p.add_argument("--roi-vessel", type=Path, default=Path("/Users/mac/PycharmProjects/pythonMPhysproject/bids_dir/mida_seg/vessel_mask.nii"))
    p.add_argument("--roi-vcsf", type=Path, default=Path("/Users/mac/PycharmProjects/pythonMPhysproject/bids_dir/mida_seg/vcsf_mask.nii"))
    p.add_argument("--roi-cgm", type=Path, default=Path("/Users/mac/PycharmProjects/pythonMPhysproject/bids_dir/mida_seg/cgm_mask.nii"))
    p.add_argument("--roi-sgm", type=Path, default=Path("/Users/mac/PycharmProjects/pythonMPhysproject/bids_dir/mida_seg/sgm_mask.nii"))
    p.add_argument("--roi-wm", type=Path, default=Path("/Users/mac/PycharmProjects/pythonMPhysproject/bids_dir/mida_seg/wm_mask.nii"))

    p.add_argument(
        "--s0-json",
        type=Path,
        default=Path("/Users/mac/PycharmProjects/pythonMPhysproject/s0_by_roi.json"),
        help="Path to s0_by_roi.json",
    )
    p.add_argument("--s0-stat", type=str, default="median", choices=["median", "mean"])

    p.add_argument("--mask", type=Path, default=None)

    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("/Users/mac/PycharmProjects/pythonMPhysproject/hv_dist/ode_sim_two"),
        help="Root output directory",
    )
    p.add_argument("--bold-prefix", type=str, default="bold_2.5mm_rep")
    p.add_argument("--save-lowres-maps", action="store_true")
    p.add_argument("--save-clean-bold", action="store_true")
    p.add_argument("--save-noise-sd-map", action="store_true")

    p.add_argument("--target-voxel-mm", type=float, default=2.5)
    p.add_argument(
        "--preblur-fwhm-mm",
        type=float,
        default=3.0,
        help="Kept for compatibility. In this version it is used as spatial smoothing FWHM applied right before noise, not before downsampling.",
    )

    p.add_argument("--tr", type=float, default=1.55)
    p.add_argument("--target-n-time", type=int, default=517,
                   help="Number of output timepoints in simulated BOLD")
    p.add_argument("--extra-pre", type=int, default=31)
    p.add_argument("--extra-post", type=int, default=93)
    p.add_argument("--internal-dt", type=float, default=1.55)

    p.add_argument("--T-sec", type=float, default=15.0)

    p.add_argument("--n-reps", type=int, default=1)
    p.add_argument("--tcnr-list", type=str, default="5.0")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--chunk-vox", type=int, default=50000,
                   help="Voxel chunk size for chunked clean construction and noise")
    p.add_argument("--out-ext", type=str, default=".nii", choices=[".nii", ".nii.gz"],
                   help="Output extension; .nii is faster than .nii.gz")

    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    tcnr_values = parse_tcnr_list(args.tcnr_list)

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    cvr_img_hr = load_nii(args.cvr_mag)
    delay_img_hr = load_nii(args.delay)

    cvr_hr = get_data(cvr_img_hr)
    delay_hr = get_data(delay_img_hr)
    ensure_same_shape(cvr_hr, delay_hr, "cvr_hr", "delay_hr")

    roi_ref_img, roi_masks = load_roi_masks(
        vessel=args.roi_vessel,
        vcsf=args.roi_vcsf,
        cgm=args.roi_cgm,
        sgm=args.roi_sgm,
        wm=args.roi_wm,
    )
    ensure_same_shape(cvr_hr, get_data(roi_ref_img), "cvr_hr", "roi_ref")

    roi_masks = apply_previous_vessel_fill_setting(
        roi_masks,
        vessel_close_iter=2,
        brain_dilate_iter=2,
    )

    s0_meta = load_s0_json(args.s0_json)
    s0_vals = choose_s0_values(s0_meta, stat=args.s0_stat)
    priority = s0_meta.get("priority", ["vessel", "vcsf", "cgm", "sgm", "wm"])
    exclusive_masks = build_exclusive_roi_masks(roi_masks, priority=priority)
    s0_hr = build_s0_map_from_rois(exclusive_masks, s0_vals)

    roi_union_mask = np.zeros_like(cvr_hr, dtype=bool)
    for roi in ROI_KEYS:
        roi_union_mask |= exclusive_masks[roi]

    if args.mask is not None:
        mask_img_hr = load_nii(args.mask)
        extra_mask_hr = get_data(mask_img_hr) > 0
        ensure_same_shape(cvr_hr, extra_mask_hr.astype(np.float32), "cvr_hr", "extra_mask_hr")
        mask_hr = roi_union_mask & extra_mask_hr
    else:
        mask_hr = roi_union_mask

    cvr_hr = cvr_hr.astype(np.float32)
    delay_hr = delay_hr.astype(np.float32)
    s0_hr = s0_hr.astype(np.float32)

    cvr_hr[~mask_hr] = 0.0
    delay_hr[~mask_hr] = 0.0
    s0_hr[~mask_hr] = 0.0

    s0_img_hr = nib.Nifti1Image(s0_hr, cvr_img_hr.affine, cvr_img_hr.header)
    mask_img_hr_float = nib.Nifti1Image(mask_hr.astype(np.float32), cvr_img_hr.affine, cvr_img_hr.header)

    cvr_img_lr = downsample_map_to_voxel_size(
        cvr_img_hr,
        order=1,
        voxel_size_mm=args.target_voxel_mm,
    )
    delay_img_lr = downsample_map_to_voxel_size(
        delay_img_hr,
        order=1,
        voxel_size_mm=args.target_voxel_mm,
    )
    s0_img_lr = downsample_map_to_voxel_size(
        s0_img_hr,
        order=1,
        voxel_size_mm=args.target_voxel_mm,
    )
    mask_img_lr = downsample_map_to_voxel_size(
        mask_img_hr_float,
        order=0,
        voxel_size_mm=args.target_voxel_mm,
    )

    cvr_lr = get_data(cvr_img_lr)
    delay_lr = get_data(delay_img_lr)
    s0_lr = get_data(s0_img_lr)
    mask_lr = get_data(mask_img_lr) > 0.5

    cvr_lr[~mask_lr] = 0.0
    delay_lr[~mask_lr] = 0.0
    s0_lr[~mask_lr] = 0.0

    long_etco2 = build_long_etco2(args.extra_pre, args.extra_post)

    required_duration_sec = int(np.ceil(args.target_n_time * args.tr + args.tr))
    scan_etco2_1hz = crop_scan_etco2(
        long_etco2,
        args.extra_pre,
        args.extra_post,
        min_length=required_duration_sec,
        pad_value=40.0,
    )

    if abs(args.internal_dt - 1.0) > 1e-6:
        t_1hz = np.arange(scan_etco2_1hz.size, dtype=np.float32)
        t_internal = np.arange(
            0.0,
            (scan_etco2_1hz.size - 1) + 1e-6,
            args.internal_dt,
            dtype=np.float32,
        )
        scan_etco2_internal = resample_1d(scan_etco2_1hz, t_1hz, t_internal)
    else:
        t_internal = np.arange(scan_etco2_1hz.size, dtype=np.float32)
        scan_etco2_internal = scan_etco2_1hz.copy()

    etco2_baseline = float(np.mean(scan_etco2_internal[:max(1, min(100, scan_etco2_internal.size))]))
    u_delta = scan_etco2_internal - etco2_baseline

    nt_out = int(args.target_n_time)
    t_out = np.arange(1, nt_out + 1, dtype=np.float32) * float(args.tr)

    if t_out[-1] > t_internal[-1]:
        raise RuntimeError(
            f"Requested output time grid extends beyond internal EtCO2 grid: "
            f"t_out[-1]={t_out[-1]:.3f}s vs t_internal[-1]={t_internal[-1]:.3f}s. "
            f"Increase padding or target duration."
        )

    print("Building lag-specific unit responses...")
    cvr_frac_flat = (cvr_lr / 100.0).reshape(-1).astype(np.float32)
    delay_flat = delay_lr.reshape(-1).astype(np.float32)
    s0_flat = s0_lr.reshape(-1).astype(np.float32)
    mask_flat = mask_lr.reshape(-1)

    lag_steps_flat, q_out_dict = build_lag_response_dict(
        u_delta=u_delta,
        tau_flat_sec=delay_flat,
        internal_dt=args.internal_dt,
        T_sec=args.T_sec,
        t_internal=t_internal,
        t_out=t_out,
    )

    print(f"Unique delay lags: {len(q_out_dict)}")
    print(f"Target output timepoints: {nt_out}")
    print(f"Output TR: {args.tr}")
    print(f"Output duration: {t_out[-1]:.3f} s")

    print("Constructing clean low-resolution BOLD...")
    bold4d_clean = construct_clean_bold_chunked(
        cvr_frac_flat=cvr_frac_flat,
        s0_flat=s0_flat,
        lag_steps_flat=lag_steps_flat,
        q_out_dict=q_out_dict,
        mask_flat=mask_flat,
        out_shape_3d=cvr_lr.shape,
        nt_out=nt_out,
        chunk_vox=args.chunk_vox,
    )

    lowres_vox_mm = voxel_sizes_from_affine(cvr_img_lr.affine)
    bold4d_for_noise = smooth_4d_spatial_only(
        bold4d_clean,
        voxel_sizes_mm=lowres_vox_mm,
        fwhm_mm=args.preblur_fwhm_mm,
        mask=mask_lr,
    )

    if args.save_lowres_maps:
        save_nii(cvr_lr, cvr_img_lr, out_root / f"gt_cvr_mag_2p5mm{args.out_ext}")
        save_nii(delay_lr, delay_img_lr, out_root / f"gt_delay_2p5mm{args.out_ext}")
        save_nii(s0_lr, s0_img_lr, out_root / f"gt_s0_2p5mm{args.out_ext}")
        save_nii(mask_lr.astype(np.float32), mask_img_lr, out_root / f"gt_mask_2p5mm{args.out_ext}")

    if args.save_clean_bold:
        save_nii(bold4d_clean, cvr_img_lr, out_root / f"bold_2p5mm_clean{args.out_ext}", tr=args.tr)

    meta = {
        "cvr_mag_hr": str(args.cvr_mag),
        "delay_hr": str(args.delay),
        "s0_json": str(args.s0_json),
        "s0_stat": args.s0_stat,
        "priority": priority,
        "s0_values_used": s0_vals,
        "fill_missing_as_vessel_mode": "previous_setting",
        "vessel_close_iter": 2,
        "brain_dilate_iter": 2,
        "target_voxel_mm": args.target_voxel_mm,
        "smoothing_before_noise_fwhm_mm": args.preblur_fwhm_mm,
        "tr": args.tr,
        "target_n_time": int(args.target_n_time),
        "extra_pre": args.extra_pre,
        "extra_post": args.extra_post,
        "internal_dt": args.internal_dt,
        "T_sec": args.T_sec,
        "n_reps": args.n_reps,
        "tcnr_values": tcnr_values,
        "seed": args.seed,
        "etco2_baseline": etco2_baseline,
        "lowres_shape": list(cvr_lr.shape),
        "n_time_output": int(nt_out),
        "n_unique_lags": int(len(q_out_dict)),
        "chunk_vox": int(args.chunk_vox),
        "required_duration_sec": int(required_duration_sec),
        "actual_internal_duration_sec": float(t_internal[-1]),
        "actual_output_duration_sec": float(t_out[-1]),
    }
    with open(out_root / "simulation_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Generating noisy repetitions...")
    for tcnr in tcnr_values:
        subdir = out_root / f"tCNR_{tcnr:.3f}"
        subdir.mkdir(parents=True, exist_ok=True)

        noise_sd = compute_noise_sd_from_tcnr(
            bold4d_clean=bold4d_for_noise,
            t_out=t_out,
            mask=mask_lr,
            tcnr=tcnr,
        )

        if args.save_noise_sd_map:
            save_nii(noise_sd, cvr_img_lr, subdir / f"noise_sd_map{args.out_ext}")

        for rep in range(args.n_reps):
            bold_rep = add_noise_repetition_chunked(
                bold4d_clean=bold4d_for_noise,
                noise_sd=noise_sd,
                mask=mask_lr,
                rng=rng,
                chunk_vox=args.chunk_vox,
            )
            out_path = subdir / f"{args.bold_prefix}{rep:04d}{args.out_ext}"
            save_nii(bold_rep, cvr_img_lr, out_path, tr=args.tr)

            if (rep + 1) % 10 == 0 or rep == 0 or rep == args.n_reps - 1:
                print(f" tCNR={tcnr:.3f} | saved {rep + 1}/{args.n_reps}")

    print(f"Done. Outputs saved under: {out_root}")


if __name__ == "__main__":
    main()