
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib

try:
    from nilearn.image import resample_img
except ImportError as e:
    raise SystemExit("Need nilearn: pip install nilearn") from e

try:
    from scipy.ndimage import gaussian_filter
except ImportError as e:
    raise SystemExit("Need scipy: pip install scipy") from e


def build_long_etco2(extra_pre, extra_post):
    parts = [
        40.0 * np.ones(extra_pre, dtype=np.float32),
        40.0 * np.ones(120, dtype=np.float32),
        50.0 * np.ones(180, dtype=np.float32),
        40.0 * np.ones(120, dtype=np.float32),
        50.0 * np.ones(180, dtype=np.float32),
        40.0 * np.ones(121, dtype=np.float32),
        40.0 * np.ones(extra_post, dtype=np.float32),
    ]
    return np.concatenate(parts)


def matlab_interp_tr(signal_1hz, tr):
    L = signal_1hz.shape[0]
    if L < 2:
        return signal_1hz.copy()
    x = np.arange(1, L + 1, dtype=np.float32)
    xi = np.arange(tr, (L - 1) + 1e-6, tr, dtype=np.float32)
    return np.interp(xi, x, signal_1hz).astype(np.float32)


def p_tr_for_delay_seconds(delay_sec, long_etco2_1hz, tr, extra_pre, extra_post):

    et = long_etco2_1hz[extra_pre: len(long_etco2_1hz) - extra_post].astype(np.float32)

    if delay_sec != 0:
        n_base = min(100, et.shape[0])
        baseline = float(et[:n_base].mean()) if n_base > 0 else float(et[0])
        if delay_sec > 0:
            et = np.concatenate([np.full(delay_sec, baseline, dtype=np.float32), et[:-delay_sec]])
        else:
            dd = abs(delay_sec)
            et = np.concatenate([et[dd:], np.full(dd, baseline, dtype=np.float32)])

    et_tr = matlab_interp_tr(et, tr)
    p_tr = (et_tr - float(long_etco2_1hz[0])).astype(np.float32)
    pmax = float(np.max(np.abs(p_tr))) if p_tr.size else 0.0
    return p_tr, pmax


def fwhm_to_sigma_vox(fwhm_mm, voxel_sizes_mm):
    sigma_mm = float(fwhm_mm) / 2.354820045
    return tuple(sigma_mm / float(v) for v in voxel_sizes_mm)

def target_affine_iso(ref_affine, voxel_mm):
    aff = np.array(ref_affine, dtype=float, copy=True)
    R = aff[:3, :3]
    norms = np.linalg.norm(R, axis=0)
    norms[norms == 0] = 1.0
    R_unit = R / norms
    aff[:3, :3] = R_unit * float(voxel_mm)
    return aff

def blur_then_resample_continuous(data_hi, ref_hi_img, fwhm_mm, target_affine, target_shape):
    if fwhm_mm <= 0:
        img_hi = nib.Nifti1Image(data_hi.astype(np.float32), affine=ref_hi_img.affine)
        img_lo = resample_img(img_hi, target_affine=target_affine, target_shape=target_shape, interpolation="continuous")
        return img_lo.get_fdata(dtype=np.float32)

    vox_hi = ref_hi_img.header.get_zooms()[:3]
    sigma_vox = fwhm_to_sigma_vox(fwhm_mm, vox_hi)
    blurred = gaussian_filter(data_hi.astype(np.float32), sigma=sigma_vox, mode="constant", cval=0.0).astype(np.float32)

    img_hi = nib.Nifti1Image(blurred, affine=ref_hi_img.affine)
    img_lo = resample_img(img_hi, target_affine=target_affine, target_shape=target_shape, interpolation="continuous")
    return img_lo.get_fdata(dtype=np.float32)

def resample_nearest_float(data_hi, ref_hi_img, target_affine, target_shape):
    img_hi = nib.Nifti1Image(data_hi.astype(np.float32), affine=ref_hi_img.affine)
    img_lo = resample_img(img_hi, target_affine=target_affine, target_shape=target_shape, interpolation="nearest")
    return img_lo.get_fdata(dtype=np.float32)

def resample_nearest_mask(mask_hi, ref_hi_img, target_affine, target_shape):
    img = nib.Nifti1Image(mask_hi.astype(np.uint8), affine=ref_hi_img.affine)
    lo = resample_img(img, target_affine=target_affine, target_shape=target_shape, interpolation="nearest")
    return (lo.get_fdata(dtype=np.float32) > 0.5)


TISSUES = ["NAWM", "CGM", "SGM", "VCSF", "VESSELS"]

DEFAULT_BASELINES = {
    "NAWM": 300.0,
    "CGM":  450.0,
    "SGM":  400.0,
    "VCSF": 600.0,
    "VESSELS": 150.0,
}

DEFAULT_TCNRS = {
    "NAWM": 0.31,
    "CGM":  1.33,
    "SGM":  1.24,
    "VCSF": 0.11,
    "VESSELS": 2.28,
}

def parse_kv_args(items):
    out: Dict[str, str] = {}
    for it in items:
        k, v = it.split("=", 1)
        out[k.strip().upper()] = v.strip()
    return out

def build_map_from_masks(shape, mask_paths, values, priority):

    out = np.zeros(shape, dtype=np.float32)
    assigned = np.zeros(shape, dtype=bool)

    for t in priority:
        t = t.upper()
        if t not in mask_paths:
            continue
        m = nib.load(mask_paths[t]).get_fdata(dtype=np.float32) > 0.5
        to_set = m & (~assigned)
        if np.any(to_set):
            out[to_set] = float(values.get(t, 0.0))
            assigned[to_set] = True

    return out, assigned


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--cvr_mag", required=True, help="High-res CVR magnitude (%/mmHg).")
    p.add_argument("--cvr_delay", required=True, help="High-res CVR delay (s).")
    p.add_argument("--outdir", required=True)

    p.add_argument("--tr", type=float, default=1.55)
    p.add_argument("--extra_pre", type=int, default=93)
    p.add_argument("--extra_post", type=int, default=31)

    p.add_argument("--down_vox", type=float, default=2.5)

    p.add_argument("--cvr_fwhm_mm", type=float, default=4.0,
                   help="FWHM (mm) to smooth CVR magnitude BEFORE resampling. 0 disables.")

    p.add_argument("--s0_fwhm_mm", type=float, default=0.0,
                   help="FWHM (mm) to smooth baseline S0 BEFORE resampling (helps anatomical look).")

    p.add_argument("--tcnr_global_scale", type=float, nargs="+", required=True,
                   help="Global multiplier(s) for tissue tCNR map. Larger -> less noise. Smaller -> more noise. "
                        "Try e.g. 1.0, 0.7, 0.5.")
    p.add_argument("--n_reps", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--mag_eps", type=float, default=0.001)

    p.add_argument("--tissue_masks", nargs="*", default=[],
                   help="Key=path. e.g. NAWM=wm.nii CGM=cgm.nii SGM=sgm.nii VCSF=vcsf.nii VESSELS=vessel.nii")
    p.add_argument("--priority", type=str, default="VESSELS,VCSF,CGM,SGM,NAWM")

    p.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--save_qc_maps", action="store_true")

    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mag_hi_img = nib.load(args.cvr_mag)
    delay_hi_img = nib.load(args.cvr_delay)
    mag_hi = mag_hi_img.get_fdata(dtype=np.float32)
    delay_hi = delay_hi_img.get_fdata(dtype=np.float32)

    mag_eps = float(args.mag_eps)
    brain_mask_hi = np.isfinite(mag_hi) & (np.abs(mag_hi) > mag_eps) & np.isfinite(delay_hi)
    mag_hi = np.where(brain_mask_hi, mag_hi, 0.0).astype(np.float32)
    delay_hi = np.where(brain_mask_hi, delay_hi, 0.0).astype(np.float32)

    print("Hi-res shape:", mag_hi.shape)
    print("Hi-res mask fraction:", float(brain_mask_hi.mean()))

    low_aff = target_affine_iso(mag_hi_img.affine, float(args.down_vox))
    mag_lo_tmp = resample_img(mag_hi_img, target_affine=low_aff, interpolation="continuous")
    low_shape = mag_lo_tmp.shape
    print("Low-res shape:", low_shape)

    mag_lo = blur_then_resample_continuous(
        mag_hi, mag_hi_img, float(args.cvr_fwhm_mm), low_aff, low_shape
    )
    delay_lo = resample_nearest_float(delay_hi, delay_hi_img, low_aff, low_shape)

    brain_mask_lo = np.isfinite(mag_lo) & (np.abs(mag_lo) > mag_eps) & np.isfinite(delay_lo)
    mag_lo = np.where(brain_mask_lo, mag_lo, 0.0).astype(np.float32)
    delay_lo = np.where(brain_mask_lo, delay_lo, 0.0).astype(np.float32)

    mask_paths = parse_kv_args(args.tissue_masks)
    if not mask_paths:
        raise SystemExit("You must provide --tissue_masks to use tissue-specific baselines and tCNRs.")

    priority = [x.strip().upper() for x in args.priority.split(",") if x.strip()]

    s0_hi, tissue_any_hi = build_map_from_masks(mag_hi.shape, mask_paths, DEFAULT_BASELINES, priority)
    tcnr_hi, _ = build_map_from_masks(mag_hi.shape, mask_paths, DEFAULT_TCNRS, priority)

    s0_lo = blur_then_resample_continuous(s0_hi, mag_hi_img, float(args.s0_fwhm_mm), low_aff, low_shape)

    tissue_any_lo = resample_nearest_mask(tissue_any_hi.astype(np.uint8), mag_hi_img, low_aff, low_shape)
    sim_mask = brain_mask_lo & tissue_any_lo

    tcnr_lo = resample_nearest_float(tcnr_hi, mag_hi_img, low_aff, low_shape)
    tcnr_lo = np.where(sim_mask, tcnr_lo, 0.0).astype(np.float32)
    tcnr_lo = np.where(sim_mask & (tcnr_lo <= 0), 1.0, tcnr_lo).astype(np.float32)

    s0_lo = np.where(sim_mask, s0_lo, 0.0).astype(np.float32)

    print("Low-res sim_mask fraction:", float(sim_mask.mean()))
    if sim_mask.any():
        print("S0 stats p05/median/p95:",
              float(np.percentile(s0_lo[sim_mask], 5)),
              float(np.median(s0_lo[sim_mask])),
              float(np.percentile(s0_lo[sim_mask], 95)))
        print("tCNR tissue map stats p05/median/p95:",
              float(np.percentile(tcnr_lo[sim_mask], 5)),
              float(np.median(tcnr_lo[sim_mask])),
              float(np.percentile(tcnr_lo[sim_mask], 95)))

    k_lo = (mag_lo / 100.0).astype(np.float32)

    delay_sec = np.ceil(delay_lo).astype(np.int32)
    delay_sec = np.where(sim_mask, delay_sec, 0).astype(np.int32)
    uniq_delays = np.unique(delay_sec[sim_mask]).astype(np.int32)
    print("Unique delay groups:", int(uniq_delays.size),
          "range", int(uniq_delays.min()), "to", int(uniq_delays.max()))

    long_et = build_long_etco2(int(args.extra_pre), int(args.extra_post))

    p_by_delay: Dict[int, np.ndarray] = {}
    pmax_by_delay: Dict[int, float] = {}
    for d in uniq_delays:
        p_tr, pmax = p_tr_for_delay_seconds(int(d), long_et, float(args.tr),
                                            int(args.extra_pre), int(args.extra_post))
        p_by_delay[int(d)] = p_tr
        pmax_by_delay[int(d)] = pmax

    T = int(next(iter(p_by_delay.values())).shape[0])
    print("TR frames:", T)

    flat_mask = sim_mask.ravel()
    flat_delay = delay_sec.ravel()
    idx_by_delay: Dict[int, np.ndarray] = {}
    for d in uniq_delays:
        idx_by_delay[int(d)] = np.flatnonzero(flat_mask & (flat_delay == int(d)))

    flat_s0 = s0_lo.ravel()
    flat_k = k_lo.ravel()
    abs_k = np.abs(flat_k).astype(np.float32)
    flat_tcnr = tcnr_lo.ravel().astype(np.float32)

    disk_dtype = np.float16 if args.dtype == "float16" else np.float32

    hdr = mag_lo_tmp.header.copy()
    hdr.set_data_dtype(disk_dtype)
    hdr.set_data_shape((*low_shape, T))
    dx, dy, dz = hdr.get_zooms()[:3]
    hdr.set_zooms((float(dx), float(dy), float(dz), float(args.tr)))
    hdr["pixdim"][5:8] = 0.0

    if args.save_qc_maps:
        hdr3 = mag_lo_tmp.header.copy()
        hdr3.set_data_dtype(np.float32)
        hdr3.set_data_shape(low_shape)
        hdr3["pixdim"][5:8] = 0.0
        nib.save(nib.Nifti1Image(mag_lo.astype(np.float32), low_aff, hdr3), str(outdir / "CVR_mag_lowres.nii.gz"))
        nib.save(nib.Nifti1Image(delay_lo.astype(np.float32), low_aff, hdr3), str(outdir / "CVR_delay_lowres.nii.gz"))
        nib.save(nib.Nifti1Image(s0_lo.astype(np.float32), low_aff, hdr3), str(outdir / "S0_lowres.nii.gz"))
        nib.save(nib.Nifti1Image(tcnr_lo.astype(np.float32), low_aff, hdr3), str(outdir / "tCNRest_lowres_tissue.nii.gz"))
        nib.save(nib.Nifti1Image(sim_mask.astype(np.uint8), low_aff, hdr3), str(outdir / "sim_mask_lowres.nii.gz"))

    nvox_lo = int(flat_mask.sum())
    print("Low-res simulated voxels:", nvox_lo)

    for tcnr_scale in args.tcnr_global_scale:
        tcnr_scale = float(tcnr_scale)

        for rep in range(int(args.n_reps)):
            rng = np.random.default_rng(int(args.seed) + rep + int(round(1e6 * tcnr_scale)))

            out_path = outdir / f"bold_lowres_tissueS0_tissueTCNR_scale{tcnr_scale:.3f}_rep{rep:04d}.nii.gz"
            tmp_dat = out_path.with_suffix(out_path.suffix + ".tmpdat")
            mm = np.memmap(tmp_dat, mode="w+", dtype=disk_dtype, shape=(*low_shape, T))

            tcnr_vox = np.where(flat_mask, flat_tcnr * tcnr_scale, 0.0).astype(np.float32)
            tcnr_vox = np.where(flat_mask, np.clip(tcnr_vox, 1e-3, None), 0.0).astype(np.float32)

            vol_flat = np.zeros(int(np.prod(low_shape)), dtype=np.float32)

            for t in range(T):
                vol_flat[:] = flat_s0

                for d in uniq_delays:
                    idx = idx_by_delay[int(d)]
                    if idx.size == 0:
                        continue
                    p_t = float(p_by_delay[int(d)][t])
                    vol_flat[idx] += flat_s0[idx] * flat_k[idx] * p_t

                for d in uniq_delays:
                    idx = idx_by_delay[int(d)]
                    if idx.size == 0:
                        continue
                    pmax = float(pmax_by_delay[int(d)])
                    if pmax == 0.0:
                        continue

                    dS = flat_s0[idx] * abs_k[idx] * pmax
                    sigma = (dS / (tcnr_vox[idx] + 1e-12)).astype(np.float32)

                    n1 = rng.standard_normal(idx.size).astype(np.float32) * sigma
                    n2 = rng.standard_normal(idx.size).astype(np.float32) * sigma
                    s = vol_flat[idx]

                    vol_flat[idx] = np.sqrt((s + n1)**2 + n2**2).astype(np.float32)

                mm[..., t] = vol_flat.reshape(low_shape).astype(disk_dtype, copy=False)

                if (t + 1) % 50 == 0 or t == T - 1:
                    print(f"[tCNRscale {tcnr_scale:.3f} rep {rep:04d}] frame {t+1}/{T}")

            mm.flush()
            img = nib.Nifti1Image(mm, affine=low_aff, header=hdr)
            nib.save(img, str(out_path))

            try:
                del mm
                tmp_dat.unlink()
            except OSError:
                pass

            print("Saved:", out_path.name)

if __name__ == "__main__":
    main()
