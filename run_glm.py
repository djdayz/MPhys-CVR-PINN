
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib


def build_long_etco2(extra_seconds):

    pre, post = int(extra_seconds[0]), int(extra_seconds[1])
    parts = [
        40.0 * np.ones(pre, dtype=np.float32),
        40.0 * np.ones(120, dtype=np.float32),
        50.0 * np.ones(180, dtype=np.float32),
        40.0 * np.ones(120, dtype=np.float32),
        50.0 * np.ones(180, dtype=np.float32),
        40.0 * np.ones(121, dtype=np.float32),
        40.0 * np.ones(post, dtype=np.float32),
    ]
    return np.concatenate(parts)


def interp_regressor_to_tr(reg_1hz, TR, n_frames):

    L = reg_1hz.shape[0]
    t_src = np.arange(L, dtype=np.float32)
    t_tgt = (np.arange(1, n_frames + 1, dtype=np.float32) * float(TR))
    t_tgt = np.clip(t_tgt, t_src[0], t_src[-1])
    return np.interp(t_tgt, t_src, reg_1hz).astype(np.float32)


def shifted_regressor_window(main_regressor_1hz, extra_seconds, shift_s, TR, n_frames):

    extra_pre, extra_post = int(extra_seconds[0]), int(extra_seconds[1])
    start = extra_pre - int(shift_s)
    win_len = int(np.ceil(n_frames * float(TR))) + 1
    end = start + win_len

    if start < 0 or end > main_regressor_1hz.shape[0]:
        return np.full((n_frames,), np.nan, dtype=np.float32)

    reg_win_1hz = main_regressor_1hz[start:end].astype(np.float32)
    return interp_regressor_to_tr(reg_win_1hz, TR=TR, n_frames=n_frames)


def load_mask(mask_path, ref_img):
    if mask_path is None:
        data0 = np.asanyarray(ref_img.dataobj[..., 0])
        return np.isfinite(data0)
    mimg = nib.load(str(mask_path))
    m = np.asanyarray(mimg.dataobj) > 0
    if m.shape != ref_img.shape[:3]:
        raise ValueError(f"Mask shape {m.shape} != BOLD shape {ref_img.shape[:3]}")
    return m


def write_nii_like(ref, data_3d, out_path):
    data_3d = data_3d.astype(np.float32)

    out = nib.Nifti1Image(data_3d, affine=ref.affine)
    zooms = ref.header.get_zooms()
    if len(zooms) >= 3:
        out.header.set_zooms(zooms[:3])

    out.header.set_data_dtype(np.float32)

    try:
        out.set_qform(ref.get_qform(), code=int(ref.header["qform_code"]))
        out.set_sform(ref.get_sform(), code=int(ref.header["sform_code"]))
    except Exception:
        pass

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(out_path))


def fast_glm_variable_delay(Y, R, shifts_s, chunk_vox=20000):

    T, V = Y.shape
    S, T2 = R.shape
    assert T2 == T

    t = np.arange(1, T + 1, dtype=np.float32)
    N = np.stack([np.ones(T, dtype=np.float32), t], axis=1)

    NTN = (N.T @ N).astype(np.float32)
    NTN_inv = np.linalg.inv(NTN).astype(np.float32)
    P = (NTN_inv @ N.T).astype(np.float32)

    Rt = R.T.astype(np.float32)
    coef_r = (P @ Rt).astype(np.float32)
    Rhat = (N @ coef_r).T.astype(np.float32)
    Rres = (R.astype(np.float32) - Rhat).astype(np.float32)
    rrss = np.sum(Rres * Rres, axis=1).astype(np.float32)

    rrss = np.where(rrss > 1e-12, rrss, np.nan).astype(np.float32)

    beta2 = np.zeros((V,), dtype=np.float32)
    best_shift = np.zeros((V,), dtype=np.int32)

    for v0 in range(0, V, chunk_vox):
        v1 = min(V, v0 + chunk_vox)
        Yc = Y[:, v0:v1].astype(np.float32)

        coef_y = (P @ Yc).astype(np.float32)
        Yhat = (N @ coef_y).astype(np.float32)
        Yres = (Yc - Yhat).astype(np.float32)
        yss = np.sum(Yres * Yres, axis=0).astype(np.float32)

        dots = (Rres @ Yres).astype(np.float32)

        sse = yss[None, :] - (dots * dots) / rrss[:, None]

        best_idx = np.nanargmin(sse, axis=0)
        best_shift[v0:v1] = shifts_s[best_idx].astype(np.int32)

        beta2[v0:v1] = (dots[best_idx, np.arange(v1 - v0)] / rrss[best_idx]).astype(np.float32)

    return beta2, best_shift


def main():
    ap = argparse.ArgumentParser(description="FAST Batch CVR GLM (MATLAB run_GLM.m style) for simulated BOLD NIfTIs.")
    ap.add_argument("--in_dir", type=Path,
                    default=Path("/Users/mac/PycharmProjects/pythonMPhysproject/hv_dist/simbold"),
                    help="Folder containing simulated 4D BOLD nifti files.")
    ap.add_argument("--glob", type=str, default="bold_2.5mm_tCNR_0.500_rep*.nii.gz",
                    help="Glob pattern to match BOLD files.")
    ap.add_argument("--out_dir", type=Path,
                    default=Path("/Users/mac/PycharmProjects/pythonMPhysproject/hv_dist/glm"),
                    help="Output folder for CVR maps + ROI CSVs.")
    ap.add_argument("--TR", type=float, default=1.55, help="TR in seconds.")
    ap.add_argument("--extra_pre", type=int, default=93, help="extraSeconds(1) in seconds (before scan).")
    ap.add_argument("--extra_post", type=int, default=31, help="extraSeconds(2) in seconds (after scan).")

    ap.add_argument("--delay_type", type=str, default="variable", choices=["variable"],
                    help="Only variable delay supported here (fast).")

    ap.add_argument("--sampling_line_delay", type=float, default=0.0,
                    help="Constant added to delay map (seconds). For simulations usually 0.")
    ap.add_argument("--baseline_nvols", type=int, default=30,
                    help="Baseline computed from first N vols.")
    ap.add_argument("--baseline_mode", type=str, default="median", choices=["mean", "median"],
                    help="Use median baseline for robustness.")
    ap.add_argument("--baseline_eps", type=float, default=1e-3,
                    help="If baseline <= eps, CVR set to 0 (or NaN if --use_nan).")
    ap.add_argument("--use_nan", action="store_true",
                    help="Write NaN for invalid baseline voxels (instead of 0).")

    ap.add_argument("--brain_mask", type=Path,
                    default=Path("/Users/mac/PycharmProjects/pythonMPhysproject/hv_dist/simbold/simbold_mask.nii"),
                    help="Brain mask in BOLD space (nii/nii.gz).")
    ap.add_argument("--roi_masks", type=str, default="",
                    help="Optional ROI masks list: name=path,name=path,... (in BOLD space).")

    ap.add_argument("--chunk_vox", type=int, default=20000,
                    help="Voxel chunk size for fast GLM (lower if RAM limited).")
    ap.add_argument("--max_voxels", type=int, default=0,
                    help="If >0, randomly sample this many voxels from brain_mask (debug).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    extra_seconds = (args.extra_pre, args.extra_post)
    long_etco2 = build_long_etco2(extra_seconds)

    roi_masks: Dict[str, Path] = {}
    if args.roi_masks.strip():
        for item in args.roi_masks.split(","):
            name, p = item.split("=", 1)
            roi_masks[name.strip()] = Path(p.strip())

    bold_files = sorted(args.in_dir.glob(args.glob))
    if not bold_files:
        raise SystemExit(f"No files matched {args.glob} in {args.in_dir}")

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i, fpath in enumerate(bold_files, start=1):
        img = nib.load(str(fpath))
        n_frames = img.shape[3]
        mask = load_mask(args.brain_mask, img)

        if args.max_voxels and args.max_voxels > 0:
            idx = np.flatnonzero(mask.ravel())
            if idx.size > args.max_voxels:
                pick = rng.choice(idx, size=args.max_voxels, replace=False)
                newmask = np.zeros(mask.size, dtype=bool)
                newmask[pick] = True
                mask = newmask.reshape(mask.shape)

        mask_idx = np.flatnonzero(mask.ravel())
        n_vox = int(mask_idx.size)

        cvr_map = np.zeros(img.shape[:3], dtype=np.float32)
        delay_map = np.zeros(img.shape[:3], dtype=np.float32)

        roi_rows: List[str] = ["roi,n_vox,beta_au_per_mmHg,baseline,cvre_pct_per_mmHg,delay_s"]
        if roi_masks:
            shifts = np.arange(-args.extra_post, args.extra_pre + 1, 1, dtype=int)
            R_all = np.stack([
                shifted_regressor_window(long_etco2, extra_seconds, int(s), args.TR, n_frames)
                for s in shifts
            ], axis=0).astype(np.float32)

            t = np.arange(1, n_frames + 1, dtype=np.float32)
            N = np.stack([np.ones(n_frames, dtype=np.float32), t], axis=1)
            NTN_inv = np.linalg.inv((N.T @ N).astype(np.float32)).astype(np.float32)
            P = (NTN_inv @ N.T).astype(np.float32)

            Rt = R_all.T
            coef_r = (P @ Rt).astype(np.float32)
            Rhat = (N @ coef_r).T.astype(np.float32)
            Rres = (R_all - Rhat).astype(np.float32)
            rrss = np.sum(Rres * Rres, axis=1).astype(np.float32)
            rrss = np.where(rrss > 1e-12, rrss, np.nan).astype(np.float32)

            for rname, rpath in roi_masks.items():
                rimg = nib.load(str(rpath))
                rmask = (np.asanyarray(rimg.dataobj) > 0)
                if rmask.shape != img.shape[:3]:
                    raise ValueError(f"ROI mask {rname} shape {rmask.shape} != {img.shape[:3]}")
                ridx = np.flatnonzero(rmask.ravel())
                if ridx.size == 0:
                    roi_rows.append(f"{rname},0,nan,nan,nan,nan")
                    continue

                ts = np.zeros((n_frames,), dtype=np.float32)
                for ttt in range(n_frames):
                    vol_t = np.asanyarray(img.dataobj[..., ttt]).ravel()
                    ts[ttt] = float(np.mean(vol_t[ridx]))

                coef_y = (P @ ts).astype(np.float32)
                yhat = (N @ coef_y).astype(np.float32)
                yres = (ts - yhat).astype(np.float32)
                yss = float(np.sum(yres * yres))

                dots = (Rres @ yres).astype(np.float32)
                sse = yss - (dots * dots) / rrss
                best_idx = int(np.nanargmin(sse))
                beta = float(dots[best_idx] / rrss[best_idx])
                shift_s = int(shifts[best_idx])

                bN = max(1, min(args.baseline_nvols, n_frames))
                baseline = float(np.median(ts[:bN]) if args.baseline_mode == "median" else np.mean(ts[:bN]))
                if (not np.isfinite(baseline)) or baseline <= args.baseline_eps:
                    cvre = np.nan if args.use_nan else 0.0
                else:
                    cvre = float((beta / baseline) * 100.0)

                delay_out = float(shift_s + args.sampling_line_delay)
                roi_rows.append(f"{rname},{int(ridx.size)},{beta:.6g},{baseline:.6g},{cvre:.6g},{delay_out:.6g}")

        Y = np.zeros((n_frames, n_vox), dtype=np.float32)
        for ttt in range(n_frames):
            vol_t = np.asanyarray(img.dataobj[..., ttt]).ravel()
            Y[ttt, :] = vol_t[mask_idx]

        shifts = np.arange(-args.extra_post, args.extra_pre + 1, 1, dtype=int)
        R_all = np.stack([
            shifted_regressor_window(long_etco2, extra_seconds, int(s), args.TR, n_frames)
            for s in shifts
        ], axis=0).astype(np.float32)

        beta2, best_shift = fast_glm_variable_delay(
            Y=Y,
            R=R_all,
            shifts_s=shifts,
            chunk_vox=args.chunk_vox,
        )

        bN = max(1, min(args.baseline_nvols, n_frames))
        if args.baseline_mode == "median":
            baseline_vox = np.median(Y[:bN, :], axis=0).astype(np.float32)
        else:
            baseline_vox = np.mean(Y[:bN, :], axis=0).astype(np.float32)

        invalid = (~np.isfinite(baseline_vox)) | (baseline_vox <= float(args.baseline_eps))
        cvr = (beta2 / baseline_vox) * 100.0
        if args.use_nan:
            cvr = cvr.astype(np.float32)
            cvr[invalid] = np.nan
        else:
            cvr = cvr.astype(np.float32)
            cvr[invalid] = 0.0

        delay = (best_shift.astype(np.float32) + float(args.sampling_line_delay)).astype(np.float32)

        nx, ny, nz = img.shape[:3]

        out_flat = np.zeros((nx * ny * nz,), dtype=np.float32)
        out_flat[mask_idx] = cvr.astype(np.float32)
        cvr_map = out_flat.reshape((nx, ny, nz), order="C")

        out_flat = np.zeros((nx * ny * nz,), dtype=np.float32)
        out_flat[mask_idx] = delay.astype(np.float32)
        delay_map = out_flat.reshape((nx, ny, nz), order="C")

        base = fpath.name
        if base.endswith(".nii.gz"):
            base = base[:-7]
        elif base.endswith(".nii"):
            base = base[:-4]
        else:
            base = Path(base).stem
        
        out_mag = args.out_dir / f"{base}_mag.nii.gz"
        out_delay = args.out_dir / f"{base}_delay.nii.gz"

        if out_mag.exists() or out_delay.exists():
            raise RuntimeError(f"Warning: output files for {fpath.name} already exist, overwriting.")

        print("Saving:", out_mag.name, out_delay.name)

        write_nii_like(img, cvr_map, out_mag)
        write_nii_like(img, delay_map, out_delay)

        if roi_masks:
            csv_path = Path(str(rep_out.with_suffix("")) + "_roi_glm.csv")
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_text("\n".join(roi_rows) + "\n")

        print(f"[{i}/{len(bold_files)}] done: {fpath.name}  voxels={n_vox}  frames={n_frames}  shifts={len(shifts)}")


if __name__ == "__main__":
    main()