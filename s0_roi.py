
import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib

EXCLUDE = {
    ("sub-04", "ses-01"),
    ("sub-04", "ses-02"),
    ("sub-11", "ses-01"),
    ("sub-12", "ses-01"),
    ("sub-12", "ses-02"),
    ("sub-13", "ses-01"),
    ("sub-13", "ses-02"),
}

ROI_FILES: Dict[str, str] = {
    "cgm": "cortical_gm_mask_in_BOLD_bin.nii",
    "sgm": "subcort_gm_mask_in_BOLD_bin.nii",
    "wm": "wm_mask_in_BOLD_bin_ero1.nii",
    "vcsf": "vcsf_mask_in_BOLD_bin.nii",
    "vessel": "vessel_mask_in_BOLD_bin.nii",
}

PRIORITY = ["vessel", "vcsf", "cgm", "sgm", "wm"]


def load_mask_bool(path, thr=0.5):
    return nib.load(path.as_posix()).get_fdata(dtype=np.float32) > thr


def main():
    ap = argparse.ArgumentParser(
        description="Compute ROI-specific S0 from saved mean BOLD images using exclusive masks with priority."
    )
    ap.add_argument("--bids_dir", type=Path, required=True, help="BIDS root")
    ap.add_argument(
        "--mean_bold_rel",
        type=str,
        default="pre/boldmcf_mean_reg.nii",
        help="Relative path from sub-*/ses-* to mean BOLD NIfTI (BOLD space).",
    )
    ap.add_argument("--roi_dirname", type=str, default="roi2bold", help="ROI directory under ses-*")
    ap.add_argument("--mask_thr", type=float, default=0.5, help="Mask threshold")
    ap.add_argument("--min_vox", type=int, default=50, help="Minimum exclusive voxels to accept an ROI")
    ap.add_argument("--out_json", type=Path, default=Path("s0_by_roi.json"))
    args = ap.parse_args()

    per_run: List[dict] = []
    s0_values: Dict[str, List[float]] = {roi: [] for roi in ROI_FILES.keys()}

    subs = sorted([p for p in args.bids_dir.glob("sub-*") if p.is_dir()])

    for sub_dir in subs:
        sub = sub_dir.name
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            ses = ses_dir.name
            if (sub, ses) in EXCLUDE:
                continue

            mean_path = ses_dir / args.mean_bold_rel
            if not mean_path.exists():
                continue

            roi_dir = ses_dir / args.roi_dirname
            if not roi_dir.exists():
                continue

            mean_img = nib.load(mean_path.as_posix())
            mean_vol = mean_img.get_fdata(dtype=np.float32)

            masks: Dict[str, np.ndarray] = {}
            for roi, fname in ROI_FILES.items():
                mpath = roi_dir / fname
                if mpath.exists():
                    masks[roi] = load_mask_bool(mpath, thr=args.mask_thr)

            if not masks:
                continue

            taken = np.zeros(mean_vol.shape, dtype=bool)
            excl: Dict[str, np.ndarray] = {}
            for roi in PRIORITY:
                if roi not in masks:
                    continue
                m_excl = masks[roi] & (~taken)
                excl[roi] = m_excl
                taken |= m_excl

            run_rec = {"sub": sub, "ses": ses, "mean_bold": str(mean_path), "S0": {}}

            for roi in PRIORITY:
                if roi not in excl:
                    continue
                nvox = int(excl[roi].sum())
                if nvox < args.min_vox:
                    continue
                s0 = float(mean_vol[excl[roi]].mean())
                run_rec["S0"][roi] = {"value": s0, "nvox_exclusive": nvox}
                s0_values[roi].append(s0)

            if run_rec["S0"]:
                per_run.append(run_rec)
                pretty = " ".join(
                    [f"{r}={run_rec['S0'][r]['value']:.2f}(n={run_rec['S0'][r]['nvox_exclusive']})"
                     for r in run_rec["S0"].keys()]
                )
                print(f"Processed {sub}/{ses} | {pretty}")

    agg = {}
    for roi, vals in s0_values.items():
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            agg[roi] = {"n_runs": 0, "median": None, "mean": None, "std": None}
        else:
            agg[roi] = {
                "n_runs": int(arr.size),
                "median": float(np.median(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=0)),
            }

    out = {
        "definition": "S0_ROI_run = mean(mean_BOLD_image[exclusive_ROI_mask]) with priority vessel>vcsf>cgm>sgm>wm; aggregate across runs.",
        "exclude": sorted(list(EXCLUDE)),
        "roi_files": ROI_FILES,
        "priority": PRIORITY,
        "mean_bold_rel": args.mean_bold_rel,
        "per_run": per_run,
        "aggregate": agg,
        "s0_by_roi_median": {roi: agg[roi]["median"] for roi in ROI_FILES.keys()},
        "s0_by_roi_mean": {roi: agg[roi]["mean"] for roi in ROI_FILES.keys()},
    }

    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {args.out_json}")

    print("\nUse these (median) as --s0_by_roi in Script 2:")
    for roi in ["wm", "sgm", "cgm", "vcsf", "vessel"]:
        med = out["s0_by_roi_median"].get(roi)
        if med is None:
            print(f"  {roi}=MISSING")
        else:
            print(f"  {roi}={med:.3f}")


if __name__ == "__main__":
    main()