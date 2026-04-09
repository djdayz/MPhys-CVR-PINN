
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib


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


def interp_regressor_to_tr(reg_1hz, TR, n_frames):
    L = reg_1hz.shape[0]
    t_src = np.arange(L, dtype=np.float32)
    t_tgt = (np.arange(1, n_frames + 1, dtype=np.float32) * float(TR))
    t_tgt = np.clip(t_tgt, t_src[0], t_src[-1])
    return np.interp(t_tgt, t_src, reg_1hz).astype(np.float32)


def shifted_regressor_window(main_regressor_1hz, extra_seconds, shift_s, TR, n_frames):
    extra_pre, _extra_post = int(extra_seconds[0]), int(extra_seconds[1])
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(out_path))


def nifti_base_name(p):
    name = p.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return p.stem


def fast_glm_variable_delay(Y, R, shifts_s, chunk_vox=20000):
    T, V = Y.shape
    S, T2 = R.shape
    if T2 != T:
        raise ValueError(f"R has T={T2} but Y has T={T}")

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
    ap = argparse.ArgumentParser(
        description="Batch CVR GLM (MATLAB variable-delay using full extraSeconds search) with fast skipping + regressor cache."
    )
    ap.add_argument("--in_dir", type=Path, required=True)
    ap.add_argument("--glob", type=str, default="bold_2.5mm_tCNR_0.500_rep*.nii.gz")
    ap.add_argument("--out_dir", type=Path, required=True)

    ap.add_argument("--TR", type=float, default=1.55)
    ap.add_argument("--extra_pre", type=int, default=93)
    ap.add_argument("--extra_post", type=int, default=31)

    ap.add_argument("--brain_mask", type=Path, required=True)

    ap.add_argument("--baseline_nvols", type=int, default=30)
    ap.add_argument("--baseline_mode", choices=["mean", "median"], default="median")
    ap.add_argument("--baseline_eps", type=float, default=1e-3)
    ap.add_argument("--use_nan", action="store_true")

    ap.add_argument("--sampling_line_delay", type=float, default=0.0)

    ap.add_argument("--chunk_vox", type=int, default=20000)
    ap.add_argument("--overwrite", action="store_true",
                    help="If set, recompute even if outputs exist.")
    ap.add_argument("--verbose_skip", action="store_true",
                    help="Print a message when skipping an existing file.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    extra_seconds = (args.extra_pre, args.extra_post)
    long_etco2 = build_long_etco2(extra_seconds)

    shifts = np.arange(-args.extra_post, args.extra_pre + 1, 1, dtype=int)
    print(f"[INFO] delay search range: {shifts.min()}..{shifts.max()} s (S={len(shifts)})")

    bold_files = sorted(args.in_dir.glob(args.glob))
    if not bold_files:
        raise SystemExit(f"No files matched {args.glob} in {args.in_dir}")

    R_cache: Dict[int, np.ndarray] = {}

    for i, fpath in enumerate(bold_files, start=1):
        base = nifti_base_name(fpath)
        out_mag = args.out_dir / f"{base}_mag.nii.gz"
        out_del = args.out_dir / f"{base}_delay.nii.gz"

        if (out_mag.exists() and out_del.exists()) and not args.overwrite:
            if args.verbose_skip:
                print(f"[{i}/{len(bold_files)}] skip (exists): {base}")
            continue

        img = nib.load(str(fpath))
        n_frames = int(img.shape[3])

        mask = load_mask(args.brain_mask, img)
        mask_idx = np.flatnonzero(mask.ravel())
        n_vox = int(mask_idx.size)
        if n_vox == 0:
            print(f"[{i}/{len(bold_files)}] WARNING empty mask: {fpath.name}")
            continue

        if n_frames not in R_cache:
            R_all = np.stack([
                shifted_regressor_window(long_etco2, extra_seconds, int(s), args.TR, n_frames)
                for s in shifts
            ], axis=0).astype(np.float32)
            R_cache[n_frames] = R_all
            print(f"[INFO] cached R_all for n_frames={n_frames} (S={R_all.shape[0]})")
        else:
            R_all = R_cache[n_frames]

        data4d = np.asanyarray(img.dataobj).astype(np.float32)
        Y = data4d.reshape(-1, n_frames)[mask_idx, :].T

        beta2, best_shift = fast_glm_variable_delay(Y, R_all, shifts, chunk_vox=args.chunk_vox)

        bN = max(1, min(args.baseline_nvols, n_frames))
        if args.baseline_mode == "median":
            baseline_vox = np.median(Y[:bN, :], axis=0).astype(np.float32)
        else:
            baseline_vox = np.mean(Y[:bN, :], axis=0).astype(np.float32)

        invalid = (~np.isfinite(baseline_vox)) | (baseline_vox <= float(args.baseline_eps))
        cvr = (beta2 / baseline_vox) * 100.0
        cvr = cvr.astype(np.float32)
        if args.use_nan:
            cvr[invalid] = np.nan
        else:
            cvr[invalid] = 0.0

        delay = (best_shift.astype(np.float32) + float(args.sampling_line_delay)).astype(np.float32)

        nx, ny, nz = img.shape[:3]
        flat_len = nx * ny * nz

        out_flat = np.zeros(flat_len, dtype=np.float32)
        out_flat[mask_idx] = cvr
        cvr_map = out_flat.reshape((nx, ny, nz), order="C")

        out_flat = np.zeros(flat_len, dtype=np.float32)
        out_flat[mask_idx] = delay
        delay_map = out_flat.reshape((nx, ny, nz), order="C")

        write_nii_like(img, cvr_map, out_mag)
        write_nii_like(img, delay_map, out_del)

        print(f"[{i}/{len(bold_files)}] done: {fpath.name}  voxels={n_vox}  frames={n_frames}  shifts={len(shifts)}")


if __name__ == "__main__":
    main()