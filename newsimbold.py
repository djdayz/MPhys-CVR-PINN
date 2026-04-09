
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.ndimage import binary_fill_holes

import json

try:
    from nilearn.image import resample_img
except ImportError as e:
    raise SystemExit("Need nilearn: pip install nilearn") from e


PRIORITY: List[str] = ["vessel", "vcsf", "cgm", "sgm", "wm"]


def build_long_etco2(extra_seconds):
    pre, post = int(extra_seconds[0]), int(extra_seconds[1])
    return np.concatenate([
        40.0 * np.ones(pre, dtype=np.float32),
        40.0 * np.ones(120, dtype=np.float32),
        50.0 * np.ones(180, dtype=np.float32),
        40.0 * np.ones(120, dtype=np.float32),
        50.0 * np.ones(180, dtype=np.float32),
        40.0 * np.ones(121, dtype=np.float32),
        40.0 * np.ones(post, dtype=np.float32),
    ]).astype(np.float32)


def downsample_1hz_to_tr_matlab(x_1hz, tr_s):
    n = int(x_1hz.shape[0])
    xp = np.arange(1, n + 1, dtype=np.float64)
    xq = np.arange(tr_s, (n - 1) + 1e-9, tr_s, dtype=np.float64)
    return np.interp(xq, xp, x_1hz.astype(np.float64)).astype(np.float32)


def simulate_noiseless_one_voxel_matlab_exact(cvr_mag_percent, cvr_delay_s, long_etco2_1hz, s0, tr_s, extra_seconds):

    cvr_mag = float(cvr_mag_percent) / 100.0

    pre, post = int(extra_seconds[0]), int(extra_seconds[1])

    etco2 = long_etco2_1hz[pre: len(long_etco2_1hz) - post].astype(np.float32)

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
    s0_v = s0_v.astype(np.float32)
    tcnr_v = np.maximum(tcnr_v.astype(np.float32), 1e-6)

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

def sample_s0_map_from_stats(s0_map, mask, median, std, rng, dist="truncnorm", sigma_scale=1.0, clip_min=0.0):

    n = int(mask.sum())
    if n == 0:
        return
    std_eff = float(std) * float(sigma_scale)

    if std_eff <= 0:
        s0_map[mask] = float(median)
        return

    if dist == "truncnorm":
        x = rng.normal(loc=float(median), scale=std_eff, size=n).astype(np.float32)
        if clip_min is not None:
            x = np.maximum(x, float(clip_min))
        s0_map[mask] = x
        return

    if dist == "lognormal":
        m = max(float(median), 1e-6)
        mu = np.log(m)
        target_var = float(std_eff) ** 2
        s2 = np.log(1.0 + target_var / (m * m))
        sigma = float(np.sqrt(max(s2, 0.0)))
        x = rng.lognormal(mean=mu, sigma=sigma, size=n).astype(np.float32)
        s0_map[mask] = x
        return

    raise ValueError(f"Unknown dist: {dist}")

def build_roi_maps_with_priority(bids_dir, ref_lr_img, base_tcnr, tcnr_ratio, s0_by_roi, s0_stats=None, s0_dist="truncnorm", s0_sigma_scale=1.0, s0_clip_min=0.0, rng=None):

    mida_seg = bids_dir / "mida_seg"
    target_shape = ref_lr_img.shape[:3]

    taken = np.zeros(target_shape, dtype=bool)
    tcnr_map = np.zeros(target_shape, dtype=np.float32)
    s0_map = np.zeros(target_shape, dtype=np.float32)

    if rng is None:
        rng = np.random.default_rng(123)

    for roi in PRIORITY:
        mask_path = mida_seg / f"{roi}_mask.nii"
        if not mask_path.exists():
            raise SystemExit(f"Missing mask: {mask_path}")

        m = resample_mask_to_lr(mask_path, ref_lr_img)
        m_excl = m & (~taken)
        if np.any(m_excl):
            tcnr_map[m_excl] = float(base_tcnr) * float(tcnr_ratio.get(roi, 1.0))

            if s0_stats is None:
                s0_map[m_excl] = float(s0_by_roi[roi])
            else:
                med = float(s0_stats[roi]["median"])
                sd  = float(s0_stats[roi]["std"])
                sample_s0_map_from_stats(
                    s0_map=s0_map,
                    mask=m_excl,
                    median=med,
                    std=sd,
                    rng=rng,
                    dist=s0_dist,
                    sigma_scale=s0_sigma_scale,
                    clip_min=s0_clip_min,
                )
        
        print(f"{roi} resampled mask: {m.sum()} voxels, exclusive at low-res: {m_excl.sum()} voxels, union so far: {taken.sum()} voxels")

    union_mask = taken

    filled = binary_fill_holes(union_mask)
    holes = filled & (~union_mask)

    if np.any(holes):
        tcnr_map[holes] = float(base_tcnr) * float(tcnr_ratio.get("vessel", 1.0))

        if s0_stats is None:
            s0_map[holes] = float(s0_by_roi["vessel"])
        else:
            med = float(s0_stats["vessel"]["median"])
            sd  = float(s0_stats["vessel"]["std"])
            sample_s0_map_from_stats(
                s0_map=s0_map,
                mask=holes,
                median=med,
                std=sd,
                rng=rng,
                dist=s0_dist,
                sigma_scale=s0_sigma_scale,
                clip_min=s0_clip_min,
            )

        union_mask = filled
        print(f"Filled {int(holes.sum())} holes in union mask with 'vessel' values. Final union voxels: {int(union_mask.sum())}")
        
    vox_idx = np.where(union_mask.ravel())[0]

    if vox_idx.size == 0:
        print("[WARNING] Union of ROI masks is empty at low-res. Check --vox_mm and mask paths.")
    
    return union_mask.ravel(), tcnr_map.ravel()[vox_idx].astype(np.float32), s0_map.ravel()[vox_idx].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Simulate low-res BOLD (MATLAB exact) from MIDA CVR maps with ROI-specific S0 and tCNR ratios.")
    ap.add_argument("--bids_dir", type=Path, required=True, help="BIDS root containing mida_seg/")
    ap.add_argument("--mag", type=Path, default=Path("hv_dist/mida_maps/CVR_mag_mida.nii.gz"), help="MIDA CVR mag map (%/mmHg)")
    ap.add_argument("--delay", type=Path, default=Path("hv_dist/mida_maps/CVR_delay_mida.nii.gz"), help="MIDA CVR delay map (s)")
    ap.add_argument("--out_dir", type=Path, default=Path("hv_dist/simbold"), help="Output directory")
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
        "--s0_by_roi",
        nargs="*",        default=[],
        help="Absolute S0 per ROI key=value, e.g. wm=292 sgm=335 cgm=325 vcsf=280 vessel=320",
    )

    ap.add_argument("--save_tcnr_map", action="store_true")
    ap.add_argument("--save_s0_map", action="store_true")
    ap.add_argument("--save_shifted_paradigm", action="store_true")

    ap.add_argument("--s0_json", type=Path, default=Path("s0_by_roi.json"),
                help="JSON file with ROI baseline stats (e.g., median + std). If set, overrides --s0_by_roi constants.")
    ap.add_argument("--s0_sigma_scale", type=float, default=1.0,
                help="Multiply ROI std by this factor when sampling voxelwise S0.")
    ap.add_argument("--s0_dist", choices=["truncnorm", "lognormal"], default="lognormal",
                help="Distribution used to sample voxelwise S0 from ROI stats.")
    ap.add_argument("--s0_clip_min", type=float, default=0.0,
                help="Minimum allowed S0 value after sampling (useful for truncnorm).")

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tcnr_ratio = {roi: 1.0 for roi in PRIORITY}
    tcnr_ratio.update(parse_kv_pairs(args.tcnr_ratio, "--tcnr_ratio"))

    s0_stats = None
    if args.s0_json is not None and args.s0_json.exists():
        s0_stats = json.loads(args.s0_json.read_text())

    s0_by_roi = parse_kv_pairs(args.s0_by_roi, "--s0_by_roi") if args.s0_by_roi else {}

    if s0_stats is not None:
        missing = [roi for roi in PRIORITY if roi not in s0_stats]
        if missing:
            raise SystemExit(f"--s0_json missing ROI(s): {missing}. Expected keys: {PRIORITY}")
    else:
        missing = [roi for roi in PRIORITY if roi not in s0_by_roi]
        if missing:
            raise SystemExit(f"--s0_by_roi missing ROI(s): {missing}. Need values for: {PRIORITY}")

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

    xyz_shape = mag_lr.shape
    flat_len = int(np.prod(xyz_shape))

    extra = (int(args.extra_pre), int(args.extra_post))
    long_etco2 = build_long_etco2(extra)

    et_scan = long_etco2[extra[0]: len(long_etco2) - extra[1]]
    T = int(downsample_1hz_to_tr_matlab((et_scan * 0.0).astype(np.float32), tr_s=float(args.tr)).shape[0])
    out_shape = xyz_shape + (T,)

    for base_tcnr in [float(x) for x in args.tcnr]:
        union_flat, tcnr_v, s0_v = build_roi_maps_with_priority(
            bids_dir=args.bids_dir,
            ref_lr_img=mag_lr_img,
            base_tcnr=base_tcnr,
            tcnr_ratio=tcnr_ratio,
            s0_by_roi=s0_by_roi,
            s0_stats=s0_stats,
            s0_dist=args.s0_dist,
            s0_sigma_scale=args.s0_sigma_scale,
            s0_clip_min=args.s0_clip_min,
            rng=np.random.default_rng(args.seed + int(round(base_tcnr * 1000))),
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

        for rep in range(int(args.n_reps)):
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

            if base_tcnr > 0:
                bold_tv = add_noise_matlab_voxelwise_tcnr(bold_tv=bold_tv, s0_v=s0_v, tcnr_v=tcnr_v, rng=rng)

            out_flat = np.zeros((flat_len, T), dtype=np.float32)
            out_flat[vox_idx, :] = bold_tv.T
            out_4d = out_flat.reshape(out_shape)

            out_img = nib.Nifti1Image(out_4d, mag_lr_img.affine, mag_lr_img.header)
            out_img.header.set_xyzt_units("mm", "sec")
            out_img.header["pixdim"][4] = float(args.tr)

            out_path = args.out_dir / f"bold_{args.vox_mm:.1f}mm_tCNR_{base_tcnr:.3f}_rep{rep:04d}.nii.gz"
            nib.save(out_img, out_path.as_posix())
            print(f"Saved {out_path}  (vox={V}, T={T})")

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