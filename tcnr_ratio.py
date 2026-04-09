
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
    "sgm": "subcort_gm_mask_in_BOLD_bin_ero1.nii",
    "wm": "wm_mask_in_BOLD_bin_ero1.nii",
    "vcsf": "vcsf_mask_in_BOLD_bin.nii",
    "vessel": "vessel_mask_in_BOLD_bin.nii",
}

PRIORITY = ["vcsf", "vessel", "cgm", "sgm", "wm"]


def load_bool_mask(path, thr=0.5):
    return nib.load(path.as_posix()).get_fdata(dtype=np.float32) > thr


def load_tcnr(path):
    return nib.load(path.as_posix()).get_fdata(dtype=np.float32)


def robust_stat(x, mode):
    x = x[np.isfinite(np.abs(x))]
    if x.size == 0:
        return float("nan")
    if mode == "median":
        return float(np.median(np.abs(x)))
    if mode == "mean":
        return float(np.mean(np.abs(x)))
    raise ValueError("mode must be median or mean")


def main():
    ap = argparse.ArgumentParser(description="Compute ROI tCNR ratios from saved tCNR maps (tCNR_masked.nii).")
    ap.add_argument("--bids_dir", type=Path, required=True)
    ap.add_argument("--roi_dirname", type=str, default="roi2bold")
    ap.add_argument("--tcnr_rel", type=str, default="cvr/tCNR_masked.nii")
    ap.add_argument("--mask_thr", type=float, default=0.5)
    ap.add_argument("--min_vox", type=int, default=50)
    ap.add_argument("--stat", type=str, default="median", choices=["median", "mean"],
                    help="Statistic within ROI per run (median recommended)")
    ap.add_argument("--ref_roi", type=str, default="cgm", choices=list(ROI_FILES.keys()),
                    help="ROI used as ratio reference (ref=1.0)")
    ap.add_argument("--out_json", type=Path, default=Path("tcnr_ratio.json"))
    args = ap.parse_args()

    per_run: List[dict] = []
    tcnr_vals: Dict[str, List[float]] = {roi: [] for roi in ROI_FILES.keys()}

    subs = sorted([p for p in args.bids_dir.glob("sub-*") if p.is_dir()])

    for sub_dir in subs:
        sub = sub_dir.name
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            ses = ses_dir.name
            if (sub, ses) in EXCLUDE:
                continue

            tcnr_path = ses_dir / args.tcnr_rel
            if not tcnr_path.exists():
                gz = ses_dir / (args.tcnr_rel + ".gz")
                if gz.exists():
                    tcnr_path = gz
                else:
                    continue

            roi_dir = ses_dir / args.roi_dirname
            if not roi_dir.exists():
                continue

            tcnr_map = load_tcnr(tcnr_path)
            run_rec = {"sub": sub, "ses": ses, "tcnr_map": str(tcnr_path), "tCNR": {}}

            masks: Dict[str, np.ndarray] = {}
            for roi, fname in ROI_FILES.items():
                mpath = roi_dir / fname
                if mpath.exists():
                    masks[roi] = load_bool_mask(mpath, thr=args.mask_thr)

            if not masks:
                continue

            taken = np.zeros(tcnr_map.shape, dtype=bool)
            excl: Dict[str, np.ndarray] = {}
            for roi in PRIORITY:
                if roi not in masks:
                    continue
                m_excl = masks[roi] & (~taken)
                excl[roi] = m_excl
                taken |= m_excl

            for roi in PRIORITY:
                if roi not in excl:
                    continue
                nvox = int(excl[roi].sum())
                if nvox < args.min_vox:
                    continue
                vals = tcnr_map[excl[roi]]
                s = robust_stat(vals, args.stat)
                if np.isfinite(s):
                    run_rec["tCNR"][roi] = {"value": s, "nvox_exclusive": nvox}
                    tcnr_vals[roi].append(s)

            if run_rec["tCNR"]:
                per_run.append(run_rec)
                pretty = " ".join([f"{r}={run_rec['tCNR'][r]['value']:.3f}" for r in run_rec["tCNR"].keys()])
                print(f"Processed {sub}/{ses} | {pretty}")

    agg = {}
    for roi, vals in tcnr_vals.items():
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            agg[roi] = {"n_runs": 0, "median": None, "mean": None, "std": None}
        else:
            agg[roi] = {
                "n_runs": int(arr.size),
                "median": float(np.median(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=0)),
            }

    ref_val = agg[args.ref_roi][args.stat] if agg[args.ref_roi][args.stat] is not None else None
    ratios = {}
    if ref_val is None or ref_val == 0:
        ratios = {roi: None for roi in ROI_FILES.keys()}
    else:
        for roi in ROI_FILES.keys():
            v = agg[roi][args.stat]
            ratios[roi] = None if v is None else float(v / ref_val)

    out = {
        "definition": f"Per-run ROI tCNR computed from {args.tcnr_rel} using exclusive masks (priority {PRIORITY}) and {args.stat}; ratios are aggregated_{args.stat}(roi)/aggregated_{args.stat}({args.ref_roi}).",
        "exclude": sorted(list(EXCLUDE)),
        "roi_files": ROI_FILES,
        "priority": PRIORITY,
        "tcnr_rel": args.tcnr_rel,
        "stat_within_roi": args.stat,
        "ref_roi": args.ref_roi,
        "per_run": per_run,
        "aggregate": agg,
        "tcnr_ratio": ratios,
    }

    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {args.out_json}")

    print("\nUse these as --tcnr_ratio in Script 2 (ref_roi=1.0):")
    for roi in ["cgm", "sgm", "wm", "vcsf", "vessel"]:
        r = ratios.get(roi)
        if r is None:
            print(f"  {roi}=MISSING")
        else:
            print(f"  {roi}={r:.3f}")


if __name__ == "__main__":
    main()