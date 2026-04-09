

import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.io import loadmat


PRIORITY: List[str] = ["vcsf", "vessel", "cgm", "sgm", "wm"]

X_CVR = np.arange(-0.5, 2.0001, 0.01, dtype=np.float32)
X_DEL = np.arange(0, 201, 1, dtype=np.int32)


def load_weights(pdf_path):
    d = loadmat(pdf_path.as_posix())
    if "w_cvr" not in d or "w_delays" not in d:
        raise KeyError(f"{pdf_path} must contain 'w_cvr' and 'w_delays'")
    w_cvr = np.asarray(d["w_cvr"]).squeeze().astype(np.float64)
    w_del = np.asarray(d["w_delays"]).squeeze().astype(np.float64)
    if w_cvr.size != X_CVR.size:
        raise ValueError(f"{pdf_path}: w_cvr length {w_cvr.size} != {X_CVR.size}")
    if w_del.size != X_DEL.size:
        raise ValueError(f"{pdf_path}: w_delays length {w_del.size} != {X_DEL.size}")
    w_cvr = w_cvr / (w_cvr.sum() + 1e-12)
    w_del = w_del / (w_del.sum() + 1e-12)
    return w_cvr, w_del


def load_mask(mask_path, thr):
    img = nib.load(mask_path.as_posix())
    data = img.get_fdata(dtype=np.float32)
    m = data > thr
    return m, img


def sample_values(n, w_cvr, w_del, rng, delay_max=200, hard_zero_delay_tail=False):
    if delay_max < 200 and hard_zero_delay_tail:
        w_del2 = w_del.copy()
        w_del2[X_DEL > delay_max] = 0.0
        w_del2 = w_del2 / (w_del2.sum() + 1e-12)
    else:
        w_del2 = w_del

    idx_cvr = rng.choice(X_CVR.size, size=n, replace=True, p=w_cvr)
    idx_del = rng.choice(X_DEL.size, size=n, replace=True, p=w_del2)

    cvr = X_CVR[idx_cvr].astype(np.float32)
    delays = X_DEL[idx_del].astype(np.int32)
    if delay_max < 200:
        delays = np.clip(delays, 0, int(delay_max))
    return cvr, delays


def main():
    ap = argparse.ArgumentParser(description="Build MIDA CVR maps from ROI PDFs with overlap priority.")
    ap.add_argument("--bids_dir", type=Path, help="BIDS root dir (contains mida_seg/)")
    ap.add_argument("--pdf_dir", type=Path, default=Path("hv_dist/pde"), help="Directory containing pdf_{roi}.mat")
    ap.add_argument("--out_dir", type=Path, default=Path("hv_dist/mida_maps"), help="Output directory (default: bids_dir/mida_cvr)")
    ap.add_argument("--mask_thr", type=float, default=0.5, help="Mask threshold (default 0.5)")
    ap.add_argument("--seed", type=int, default=123, help="Base RNG seed (default 123)")
    ap.add_argument("--seed_by_roi", type=Path, default=None, help="Optional JSON mapping roi->seed")
    ap.add_argument("--delay_max", type=int, default=200, help="Delay cap (e.g., 95). Default 200.")
    ap.add_argument("--hard_zero_delay_tail", action="store_true",
                    help="Zero out PDF weights for delays > delay_max before sampling.")
    ap.add_argument("--background", choices=["zero", "nan"], default="zero",
                    help="Background outside all masks set to 0 or NaN. Default 0.")
    ap.add_argument("--save_union_mask", action="store_true", help="Save union mask used.")
    args = ap.parse_args()

    bids_dir: Path = args.bids_dir
    mida_seg = bids_dir / "mida_seg"
    if not mida_seg.exists():
        raise SystemExit(f"Missing directory: {mida_seg}")

    out_dir = args.out_dir if args.out_dir is not None else (bids_dir / "mida_cvr")
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_map: Dict[str, int] = {}
    if args.seed_by_roi is not None:
        seed_map = json.loads(args.seed_by_roi.read_text())

    masks: Dict[str, np.ndarray] = {}
    ref_img: Optional[nib.Nifti1Image] = None
    shape: Optional[Tuple[int, int, int]] = None

    for roi in PRIORITY:
        mask_path = mida_seg / f"{roi}_mask.nii"
        if not mask_path.exists():
            raise SystemExit(f"Missing mask: {mask_path}")
        m, img = load_mask(mask_path, thr=args.mask_thr)

        if ref_img is None:
            ref_img = img
            shape = m.shape
        else:
            if m.shape != shape:
                raise SystemExit(f"Shape mismatch for {roi}: {m.shape} vs {shape}")

        masks[roi] = m

    assert ref_img is not None and shape is not None

    taken = np.zeros(shape, dtype=bool)
    exclusive: Dict[str, np.ndarray] = {}

    for roi in PRIORITY:
        m = masks[roi]
        m_excl = m & (~taken)
        exclusive[roi] = m_excl
        taken |= m_excl

    union_mask = taken.copy()

    mag_out = np.zeros(shape, dtype=np.float32)
    del_out = np.zeros(shape, dtype=np.float32)

    stats = {
        "_priority_order": PRIORITY,
        "_delay_max": int(args.delay_max),
        "_hard_zero_delay_tail": bool(args.hard_zero_delay_tail),
    }

    for roi in PRIORITY:
        pdf_path = args.pdf_dir / f"pdf_{roi}.mat"
        if not pdf_path.exists():
            raise SystemExit(f"Missing PDF: {pdf_path}")

        w_cvr, w_del = load_weights(pdf_path)

        m_excl = exclusive[roi]
        idx = np.where(m_excl.ravel())[0]
        n = int(idx.size)
        if n == 0:
            print(f"[INFO] {roi:6s} exclusive voxels = 0 (fully overlapped or empty). Skipping.")
            stats[roi] = {"nvox_exclusive": 0}
            continue

        roi_seed = int(seed_map.get(roi, args.seed + (hash(roi) % 10_000)))
        rng = np.random.default_rng(roi_seed)

        cvr_vals, del_vals = sample_values(
            n=n,
            w_cvr=w_cvr,
            w_del=w_del,
            rng=rng,
            delay_max=int(args.delay_max),
            hard_zero_delay_tail=bool(args.hard_zero_delay_tail),
        )

        mag_flat = mag_out.ravel()
        del_flat = del_out.ravel()
        mag_flat[idx] = cvr_vals
        del_flat[idx] = del_vals.astype(np.float32)

        stats[roi] = {
            "nvox_exclusive": n,
            "seed": roi_seed,
            "cvr_mean": float(np.mean(cvr_vals)),
            "cvr_std": float(np.std(cvr_vals, ddof=0)),
            "delay_mean": float(np.mean(del_vals)),
            "delay_std": float(np.std(del_vals, ddof=0)),
        }

        print(f"[OK] {roi:6s} exclusive n={n} (wins overlaps)  CVR mean={stats[roi]['cvr_mean']:.4f}  delay mean={stats[roi]['delay_mean']:.2f}s")

    if args.background == "nan":
        mag_out[~union_mask] = np.nan
        del_out[~union_mask] = np.nan
    else:
        mag_out[~union_mask] = 0.0
        del_out[~union_mask] = 0.0

    mag_path = out_dir / "CVR_mag_mida.nii.gz"
    del_path = out_dir / "CVR_delay_mida.nii.gz"

    mag_img = nib.Nifti1Image(mag_out.astype(np.float32), ref_img.affine, ref_img.header)
    del_img = nib.Nifti1Image(del_out.astype(np.float32), ref_img.affine, ref_img.header)

    nib.save(mag_img, mag_path.as_posix())
    nib.save(del_img, del_path.as_posix())
    print(f"\nSaved {mag_path}")
    print(f"Saved {del_path}")

    if args.save_union_mask:
        um_path = out_dir / "mida_mask.nii.gz"
        um_img = nib.Nifti1Image(union_mask.astype(np.uint8), ref_img.affine, ref_img.header)
        nib.save(um_img, um_path.as_posix())
        print(f"Saved {um_path}")

    stats_path = out_dir / "mida_cvr_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Saved {stats_path}")


if __name__ == "__main__":
    main()