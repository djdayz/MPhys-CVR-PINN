
import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to

try:
    from scipy.ndimage import (
        gaussian_filter,
        binary_fill_holes,
        binary_closing,
        binary_dilation,
        label,
    )
except ImportError as e:
    raise SystemExit("Need scipy: pip install scipy") from e


PRIORITY: List[str] = ["vessel", "vcsf", "cgm", "sgm", "wm"]


def build_long_etco2(extra_seconds):
    pre, post = int(extra_seconds[0]), int(extra_seconds[1])
    return np.concatenate(
        [
            40.0 * np.ones(pre, dtype=np.float32),
            40.0 * np.ones(120, dtype=np.float32),
            50.0 * np.ones(180, dtype=np.float32),
            40.0 * np.ones(120, dtype=np.float32),
            50.0 * np.ones(180, dtype=np.float32),
            40.0 * np.ones(121, dtype=np.float32),
            40.0 * np.ones(post, dtype=np.float32),
        ]
    ).astype(np.float32)


def downsample_1hz_to_tr_matlab(x_1hz, tr_s):
    n = int(x_1hz.shape[0])
    xp = np.arange(1, n + 1, dtype=np.float64)
    xq = np.arange(tr_s, (n - 1) + 1e-9, tr_s, dtype=np.float64)
    return np.interp(xq, xp, x_1hz.astype(np.float64)).astype(np.float32)


def simulate_noiseless_one_voxel_matlab_exact(cvr_mag_percent, cvr_delay_s, long_etco2_1hz, s0, tr_s, extra_seconds):

    cvr_mag = float(cvr_mag_percent) / 100.0

    pre, post = int(extra_seconds[0]), int(extra_seconds[1])

    etco2 = long_etco2_1hz[pre : len(long_etco2_1hz) - post].astype(np.float32)

    bold_1hz = etco2 * cvr_mag

    frac_baseline = float(np.mean(bold_1hz[:100]))

    d = int(np.ceil(float(cvr_delay_s)))
    if d > 0:
        bold_1hz = np.concatenate([np.full(d, frac_baseline, np.float32), bold_1hz[:-d]])
    elif d < 0:
        dd = abs(d)
        bold_1hz = np.concatenate([bold_1hz[dd:], np.full(dd, frac_baseline, np.float32)])

    bold_tr = downsample_1hz_to_tr_matlab(bold_1hz, tr_s=float(tr_s))

    et0 = float(long_etco2_1hz[0])
    bold_tr = (float(s0) * bold_tr + float(s0) * (1.0 - cvr_mag * et0)).astype(np.float32)
    return bold_tr


def add_noise_matlab_voxelwise_tcnr(bold_tv, s0_v, tcnr_v, rng):

    T, V = bold_tv.shape
    s0_v = s0_v.astype(np.float32, copy=False)
    tcnr_v = np.maximum(tcnr_v.astype(np.float32, copy=False), 1e-6)

    dS = np.max(np.abs(bold_tv - s0_v[None, :]), axis=0).astype(np.float32)
    std_noise = dS / tcnr_v

    noise = rng.standard_normal((T, V)).astype(np.float32)
    noise = (noise - noise.mean(axis=0, keepdims=True)) / (noise.std(axis=0, keepdims=True) + 1e-12)
    noise = noise * std_noise[None, :]
    noise = noise - noise.mean(axis=0, keepdims=True)

    return (bold_tv + noise).astype(np.float32)


def resample_mask_to_lr(mask_path, ref_lr_img):
    m_img = nib.load(mask_path.as_posix())
    m_lr = resample_from_to(m_img, ref_lr_img, order=0)
    return (m_lr.get_fdata(dtype=np.float32) > 0.5)


def parse_kv_pairs(pairs, name):
    out: Dict[str, float] = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"Bad {name} entry '{p}', expected key=value")
        k, v = p.split("=", 1)
        out[k.strip().lower()] = float(v.strip())
    return out


def fwhm_to_sigma_mm(fwhm_mm):
    return float(fwhm_mm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def masked_gaussian_smooth_3d(vol3d, mask3d, sigma_vox):

    vol3d = vol3d.astype(np.float32, copy=False)
    maskf = mask3d.astype(np.float32, copy=False)

    num = gaussian_filter(vol3d * maskf, sigma=float(sigma_vox), mode="constant", cval=0.0)
    den = gaussian_filter(maskf, sigma=float(sigma_vox), mode="constant", cval=0.0)

    out = np.zeros_like(vol3d, dtype=np.float32)
    good = den > 1e-6
    out[good] = (num[good] / den[good]).astype(np.float32)
    return out


def masked_gaussian_smooth_4d(vol4d, mask3d, sigma_vox):

    X, Y, Z, T = vol4d.shape
    out = np.zeros_like(vol4d, dtype=np.float32)
    for t in range(T):
        out[..., t] = masked_gaussian_smooth_3d(vol4d[..., t], mask3d, sigma_vox=float(sigma_vox))
    return out


def _safe_get_stat(stats, roi, key):
    try:
        v = stats[roi][key]
    except Exception:
        return None
    try:
        return float(v)
    except Exception:
        return None


def sample_s0_values(rng, n, median, std, dist, sigma_scale, clip_min):

    std_eff = float(std) * float(sigma_scale)
    med = float(median)

    if n <= 0:
        return np.zeros((0,), dtype=np.float32)

    if std_eff <= 0:
        return med * np.ones(n, dtype=np.float32)

    if dist == "truncnorm":
        x = rng.normal(loc=med, scale=std_eff, size=n).astype(np.float32)
        x = np.maximum(x, float(clip_min))
        return x

    if dist == "lognormal":
        m = max(med, 1e-6)
        mu = np.log(m)

        target_var = std_eff ** 2
        s2 = np.log(1.0 + target_var / (m * m))
        sigma = float(np.sqrt(max(s2, 0.0)))
        x = rng.lognormal(mean=mu, sigma=sigma, size=n).astype(np.float32)
        x = np.maximum(x, float(clip_min))
        return x

    raise ValueError(f"Unknown s0_dist: {dist}")


def make_brain_mask_from_union(union_mask, close_iter, dilate_iter):

    m = union_mask.astype(bool)

    if close_iter > 0:
        m = binary_closing(m, iterations=int(close_iter))

    m = binary_fill_holes(m)

    lab, n = label(m)
    if n > 1:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        keep = counts.argmax()
        m = (lab == keep)

    if dilate_iter > 0:
        m = binary_dilation(m, iterations=int(dilate_iter))

    return m.astype(bool)


def build_roi_maps_with_priority(bids_dir, ref_lr_img, base_tcnr, tcnr_ratio, s0_by_roi, s0_stats, s0_dist, s0_sigma_scale, s0_clip_min, rng, fill_missing_as_vessel, brain_close_iter, brain_dilate_iter):

    mida_seg = bids_dir / "mida_seg"
    target_shape = ref_lr_img.shape[:3]

    taken = np.zeros(target_shape, dtype=bool)
    tcnr_map = np.zeros(target_shape, dtype=np.float32)
    s0_map = np.zeros(target_shape, dtype=np.float32)

    for roi in PRIORITY:
        mask_path = mida_seg / f"{roi}_mask.nii"
        if not mask_path.exists():
            raise SystemExit(f"Missing mask: {mask_path}")

        m = resample_mask_to_lr(mask_path, ref_lr_img)
        m_excl = m & (~taken)

        if np.any(m_excl):
            tcnr_map[m_excl] = float(base_tcnr) * float(tcnr_ratio.get(roi, 1.0))

            if s0_stats is not None:
                med = _safe_get_stat(s0_stats, roi, "median")
                sd = _safe_get_stat(s0_stats, roi, "std")
                if med is None:
                    med = float(s0_by_roi.get(roi, 300.0))
                if sd is None:
                    sd = 0.0
                vals = sample_s0_values(
                    rng=rng,
                    n=int(m_excl.sum()),
                    median=float(med),
                    std=float(sd),
                    dist=s0_dist,
                    sigma_scale=float(s0_sigma_scale),
                    clip_min=float(s0_clip_min),
                )
                s0_map[m_excl] = vals
            else:
                s0_map[m_excl] = float(s0_by_roi.get(roi, 300.0))

            taken |= m_excl

        print(
            f"{roi} resampled mask: {int(m.sum())} voxels, "
            f"exclusive at low-res: {int(m_excl.sum())} voxels, union so far: {int(taken.sum())} voxels"
        )

    union_mask = taken

    if fill_missing_as_vessel:
        brain_mask = make_brain_mask_from_union(
            union_mask,
            close_iter=int(brain_close_iter),
            dilate_iter=int(brain_dilate_iter),
        )
        missing = brain_mask & (~union_mask)

        if np.any(missing):
            tcnr_map[missing] = float(base_tcnr) * float(tcnr_ratio.get("vessel", 1.0))

            if s0_stats is not None:
                med = _safe_get_stat(s0_stats, "vessel", "median")
                sd = _safe_get_stat(s0_stats, "vessel", "std")
                if med is None:
                    med = float(s0_by_roi.get("vessel", 250.0))
                if sd is None:
                    sd = 0.0
                vals = sample_s0_values(
                    rng=rng,
                    n=int(missing.sum()),
                    median=float(med),
                    std=float(sd),
                    dist=s0_dist,
                    sigma_scale=float(s0_sigma_scale),
                    clip_min=float(s0_clip_min),
                )
                s0_map[missing] = vals
            else:
                s0_map[missing] = float(s0_by_roi.get("vessel", 250.0))

            union_mask = brain_mask
            print(
                f"[INFO] Brain-mask fill: assigned {int(missing.sum())} missing voxels as 'vessel'. "
                f"Final union voxels: {int(union_mask.sum())}"
            )
        else:
            union_mask = brain_mask
            print("[INFO] Brain-mask fill: no missing voxels to assign.")

    vox_idx = np.where(union_mask.ravel())[0]
    if vox_idx.size == 0:
        print("[WARNING] Union of ROI masks is empty at low-res. Check --vox_mm and mask paths.")

    return union_mask.ravel(), tcnr_map.ravel()[vox_idx].astype(np.float32), s0_map.ravel()[vox_idx].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Simulate low-res BOLD (MATLAB exact) from MIDA CVR maps with ROI-specific S0 and tCNR ratios. "
            "Includes brain-mask based filling of missing voxels as vessel + optional S0 sampling from JSON + spatial smoothing."
        )
    )
    ap.add_argument("--bids_dir", type=Path, default="/Users/mac/PycharmProjects/pythonMPhysproject/bids_dir", help="BIDS root containing mida_seg/")
    ap.add_argument("--mag", type=Path, default=Path("hv_dist/mida_maps/CVR_mag_mida.nii.gz"),
                    help="MIDA CVR mag map (%/mmHg)")
    ap.add_argument("--delay", type=Path, default=Path("hv_dist/mida_maps/CVR_delay_mida.nii.gz"),
                    help="MIDA CVR delay map (s)")
    ap.add_argument("--out_dir", type=Path, default=Path("hv_dist/glmgt"), help="Output directory")
    ap.add_argument("--tr", type=float, default=1.55, help="TR (s), e.g. 1.55")
    ap.add_argument("--vox_mm", type=float, default=2.5, help="Target isotropic voxel size (mm), default 2.5")
    ap.add_argument("--extra_pre", type=int, default=93)
    ap.add_argument("--extra_post", type=int, default=31)
    ap.add_argument("--tcnr", type=float, nargs="+", default=[1.0], help="Base tCNR scenario(s)")
    ap.add_argument("--n_reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument(
        "--tcnr_ratio",
        nargs="*",
        default=[],
        help="ROI ratios key=value, e.g. cgm=1 sgm=0.95 wm=0.8 vcsf=0.6 vessel=1.3 (missing -> 1)",
    )

    ap.add_argument(
        "--s0_json",
        type=Path,
        default=Path("s0_by_roi.json"),
        help=(
            "JSON file with ROI baseline stats. Your file has stats under key 'aggregate'. "
            "We read json['aggregate'] if present. Required fields per ROI: median, std."
        ),
    )
    ap.add_argument(
        "--s0_by_roi",
        nargs="*",
        default=[],
        help=(
            "Fallback absolute S0 per ROI key=value, e.g. wm=292 sgm=335 cgm=325 vcsf=280 vessel=320. "
            "Used if --s0_json missing or ROI stats missing."
        ),
    )
    ap.add_argument("--s0_sigma_scale", type=float, default=1.0,
                    help="Multiply ROI std by this factor when sampling voxelwise S0.")
    ap.add_argument("--s0_dist", choices=["truncnorm", "lognormal"], default="lognormal",
                    help="Distribution used to sample voxelwise S0 from ROI stats.")
    ap.add_argument("--s0_clip_min", type=float, default=0.0,
                    help="Minimum allowed S0 value after sampling.")

    ap.add_argument("--save_tcnr_map", action="store_true")
    ap.add_argument("--save_s0_map", action="store_true")
    ap.add_argument("--save_gt_maps", action="store_true",
                    help="Save downsampled low-resolution GT CVR magnitude and delay maps.")
    ap.add_argument("--save_shifted_paradigm", action="store_true")

    ap.add_argument(
        "--smooth_fwhm_mm",
        type=float,
        default=3.0,
        help="Gaussian spatial smoothing FWHM (mm) applied to noiseless low-res BOLD before adding noise. Set 0 to disable.",
    )

    ap.add_argument(
        "--fill_missing_as_vessel",
        action="store_true",
        help="Build a brain mask from union tissue masks and assign any missing voxels inside it as 'vessel'.",
    )
    ap.add_argument("--brain_close_iter", type=int, default=1,
                    help="Binary closing iterations when building brain mask.")
    ap.add_argument("--brain_dilate_iter", type=int, default=2,
                    help="Binary dilation iterations when building brain mask.")

    ap.add_argument(
        "--smooth_delta_only",
        action="store_true",
        help="If set, smooth only (BOLD - S0) then add S0 back. Prevents baseline homogenization.",
    )

    ap.add_argument("--rep_start", type=int, default=0, help="Starting repetition index (for parallelization)")

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tcnr_ratio = {roi: 1.0 for roi in PRIORITY}
    tcnr_ratio.update(parse_kv_pairs(args.tcnr_ratio, "--tcnr_ratio"))

    s0_by_roi = {roi: 300.0 for roi in PRIORITY}
    if args.s0_by_roi:
        s0_by_roi.update(parse_kv_pairs(args.s0_by_roi, "--s0_by_roi"))

    s0_stats: Optional[dict] = None
    if args.s0_json is not None:
        if not args.s0_json.exists():
            raise SystemExit(f"--s0_json not found: {args.s0_json}")
        tmp = json.loads(args.s0_json.read_text())
        s0_stats = tmp.get("aggregate", tmp)

        print("[INFO] Using s0_stats keys:", list(s0_stats.keys()))
        for k in ["cgm", "sgm", "wm", "vcsf", "vessel"]:
            if k in s0_stats:
                print(f"  {k}: median={s0_stats[k].get('median')}, std={s0_stats[k].get('std')}")

    mag_hr = nib.load(args.mag.as_posix())
    delay_hr = nib.load(args.delay.as_posix())

    zooms = mag_hr.header.get_zooms()[:3]
    scale = np.array(zooms, dtype=float) / float(args.vox_mm)
    target_shape = tuple(np.maximum(1, np.round(np.array(mag_hr.shape[:3]) * scale).astype(int)))

    A = mag_hr.affine.copy()
    R = A[:3, :3]
    dirs = R / (np.linalg.norm(R, axis=0, keepdims=True) + 1e-12)
    R_new = dirs * float(args.vox_mm)
    A_new = A.copy()
    A_new[:3, :3] = R_new

    ref_lr = nib.Nifti1Image(np.zeros(target_shape, dtype=np.float32), A_new)

    mag_lr_img = resample_from_to(mag_hr, ref_lr, order=1)
    ref_lr_img = mag_lr_img
    delay_lr_img = resample_from_to(delay_hr, ref_lr, order=1)

    mag_lr = mag_lr_img.get_fdata(dtype=np.float32)
    delay_lr = delay_lr_img.get_fdata(dtype=np.float32)

    if args.save_gt_maps:
        mag_gt_path = args.out_dir / f"CVR_mag_lowres_{args.vox_mm:.1f}mm.nii.gz"
        delay_gt_path = args.out_dir / f"CVR_delay_lowres_{args.vox_mm:.1f}mm.nii.gz"
        nib.save(nib.Nifti1Image(mag_lr, mag_lr_img.affine, mag_lr_img.header), mag_gt_path.as_posix())
        nib.save(nib.Nifti1Image(delay_lr, delay_lr_img.affine, delay_lr_img.header), delay_gt_path.as_posix())
        print(f"Saved {mag_gt_path}")
        print(f"Saved {delay_gt_path}")

    xyz_shape = mag_lr.shape
    flat_len = int(np.prod(xyz_shape))

    extra = (int(args.extra_pre), int(args.extra_post))
    long_etco2 = build_long_etco2(extra)

    et_scan = long_etco2[extra[0] : len(long_etco2) - extra[1]]
    T = int(downsample_1hz_to_tr_matlab((et_scan * 0.0).astype(np.float32), tr_s=float(args.tr)).shape[0])
    out_shape = xyz_shape + (T,)

    do_smooth = float(args.smooth_fwhm_mm) > 0.0
    if do_smooth:
        sigma_mm = fwhm_to_sigma_mm(float(args.smooth_fwhm_mm))
        sigma_vox = sigma_mm / float(args.vox_mm)
        print(f"[INFO] Spatial smoothing enabled: FWHM={args.smooth_fwhm_mm:.3f}mm -> sigma_vox={sigma_vox:.3f}")
    else:
        sigma_vox = 0.0
        print("[INFO] Spatial smoothing disabled (--smooth_fwhm_mm 0).")

    for base_tcnr in [float(x) for x in args.tcnr]:
        rng_maps = np.random.default_rng(int(args.seed) + int(round(base_tcnr * 1000)))

        union_flat, tcnr_v, s0_v = build_roi_maps_with_priority(
            bids_dir=args.bids_dir,
            ref_lr_img=ref_lr_img,
            base_tcnr=base_tcnr,
            tcnr_ratio=tcnr_ratio,
            s0_by_roi=s0_by_roi,
            s0_stats=s0_stats,
            s0_dist=args.s0_dist,
            s0_sigma_scale=args.s0_sigma_scale,
            s0_clip_min=args.s0_clip_min,
            rng=rng_maps,
            fill_missing_as_vessel=bool(args.fill_missing_as_vessel),
            brain_close_iter=int(args.brain_close_iter),
            brain_dilate_iter=int(args.brain_dilate_iter),
        )
        vox_idx = np.where(union_flat)[0]
        if vox_idx.size == 0:
            raise SystemExit("Union of ROI masks is empty at low-res.")

        if args.save_tcnr_map:
            tcnr_map = np.zeros(xyz_shape, dtype=np.float32)
            tcnr_map.ravel()[vox_idx] = tcnr_v
            p = args.out_dir / f"tCNR_{args.vox_mm:.1f}mm_base{base_tcnr:.3f}.nii.gz"
            nib.save(nib.Nifti1Image(tcnr_map, mag_lr_img.affine, mag_lr_img.header), p.as_posix())
            print(f"Saved {p}")

        if args.save_s0_map:
            s0_map = np.zeros(xyz_shape, dtype=np.float32)
            s0_map.ravel()[vox_idx] = s0_v
            p = args.out_dir / f"S0_{args.vox_mm:.1f}mm_base{base_tcnr:.3f}.nii.gz"
            nib.save(nib.Nifti1Image(s0_map, mag_lr_img.affine, mag_lr_img.header), p.as_posix())
            print(f"Saved {p}")

        mag_v = mag_lr.ravel()[vox_idx].astype(np.float32)
        delay_v = delay_lr.ravel()[vox_idx].astype(np.float32)

        union_mask_3d = union_flat.reshape(xyz_shape)

        rep_start = int(args.rep_start)
        rep_end = rep_start + int(args.n_reps)

        for rep in range(rep_start, rep_end):
            rng = np.random.default_rng(int(args.seed) + rep + int(round(base_tcnr * 1000)))

            V = mag_v.shape[0]
            bold_tv = np.empty((T, V), dtype=np.float32)
            sp_tv: Optional[np.ndarray] = np.empty((T, V), dtype=np.int32) if args.save_shifted_paradigm else None

            for i in range(V):
                b = simulate_noiseless_one_voxel_matlab_exact(
                    cvr_mag_percent=float(mag_v[i]),
                    cvr_delay_s=float(delay_v[i]),
                    long_etco2_1hz=long_etco2,
                    s0=float(s0_v[i]),
                    tr_s=float(args.tr),
                    extra_seconds=extra,
                )
                bold_tv[:, i] = b

                if sp_tv is not None:
                    shifted = b - float(s0_v[i])
                    denom = float(np.max(np.abs(shifted))) if np.max(np.abs(shifted)) != 0 else 1.0
                    sp_tv[:, i] = np.abs(np.rint(shifted / denom)).astype(np.int32)

            if do_smooth:
                tmp_flat = np.zeros((flat_len, T), dtype=np.float32)
                tmp_flat[vox_idx, :] = bold_tv.T
                tmp_4d = tmp_flat.reshape(out_shape)

                s0_map3d = np.zeros(xyz_shape, dtype=np.float32)
                s0_map3d.ravel()[vox_idx] = s0_v

                if args.smooth_delta_only:
                    delta_4d = tmp_4d - s0_map3d[..., None]
                    delta_4d_sm = masked_gaussian_smooth_4d(delta_4d, union_mask_3d, sigma_vox=float(sigma_vox))
                    tmp_4d_sm = delta_4d_sm + s0_map3d[..., None]
                    s0_v_for_noise = s0_v
                else:
                    tmp_4d_sm = masked_gaussian_smooth_4d(tmp_4d, union_mask_3d, sigma_vox=float(sigma_vox))
                    s0_map3d_sm = masked_gaussian_smooth_3d(s0_map3d, union_mask_3d, sigma_vox=float(sigma_vox))
                    s0_v_for_noise = s0_map3d_sm.ravel()[vox_idx].astype(np.float32)

                tmp_flat_sm = tmp_4d_sm.reshape(flat_len, T)
                bold_tv = tmp_flat_sm[vox_idx, :].T.astype(np.float32)
            else:
                s0_v_for_noise = s0_v

            if base_tcnr > 0:
                bold_tv = add_noise_matlab_voxelwise_tcnr(
                    bold_tv=bold_tv,
                    s0_v=s0_v_for_noise,
                    tcnr_v=tcnr_v,
                    rng=rng,
                )

            out_flat = np.zeros((flat_len, T), dtype=np.float32)
            out_flat[vox_idx, :] = bold_tv.T
            out_4d = out_flat.reshape(out_shape)

            out_img = nib.Nifti1Image(out_4d, mag_lr_img.affine, mag_lr_img.header)
            out_img.header.set_xyzt_units("mm", "sec")
            out_img.header["pixdim"][4] = float(args.tr)

            out_path = args.out_dir / f"bold_{args.vox_mm:.1f}mm_tCNR_{base_tcnr:.3f}_rep{rep:04d}.nii.gz"
            nib.save(out_img, out_path.as_posix())
            print(f"Saved {out_path} (vox={V}, T={T})")

            if sp_tv is not None:
                sp_flat = np.zeros((flat_len, T), dtype=np.int16)
                sp_flat[vox_idx, :] = sp_tv.T.astype(np.int16)
                sp_4d = sp_flat.reshape(out_shape)
                sp_img = nib.Nifti1Image(sp_4d, mag_lr_img.affine, mag_lr_img.header)
                sp_img.header["pixdim"][4] = float(args.tr)
                sp_path = args.out_dir / f"shiftedParadigm_{args.vox_mm:.1f}mm_tCNR_{base_tcnr:.3f}_rep{rep:04d}.nii.gz"
                nib.save(sp_img, sp_path.as_posix())
                print(f"Saved {sp_path}")


if __name__ == "__main__":
    main()
