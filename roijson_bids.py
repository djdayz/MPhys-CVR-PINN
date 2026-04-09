
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_nvox(qc, roi, mode):
    qc_r = qc[qc["roi"] == roi]
    if qc_r.empty:
        raise ValueError(f"No rows found in qc file for roi='{roi}'")

    s = qc_r["n_mag"].astype(float)

    if mode == "median":
        return int(np.round(np.median(s)))
    if mode == "mean":
        return int(np.round(np.mean(s)))
    if mode == "min":
        return int(np.min(s))
    if mode == "max":
        return int(np.max(s))
    raise ValueError("mode must be one of: median, mean, min, max")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc_counts_csv", required=True, help="qc_voxel_counts_per_subses.csv")
    ap.add_argument("--pooled_csv", required=True, help="pooled_empirical_distributions.csv (roi,metric,value)")
    ap.add_argument("--out_json", default="roi_config.json")
    ap.add_argument("--mode", choices=["median", "mean", "min", "max"], default="median",
                    help="How to pick n_vox per ROI across sub/ses.")
    ap.add_argument("--rois", default="cgm,sgm,wm,vcsf,vessel",
                    help="Comma-separated ROI names to include.")
    args = ap.parse_args()

    qc_path = Path(args.qc_counts_csv)
    pooled_path = Path(args.pooled_csv)

    qc = pd.read_csv(qc_path)
    pooled = pd.read_csv(pooled_path)

    qc["roi"] = qc["roi"].astype(str).str.lower().str.strip()
    pooled["roi"] = pooled["roi"].astype(str).str.lower().str.strip()
    pooled["metric"] = pooled["metric"].astype(str).str.lower().str.strip()

    rois = [r.strip().lower() for r in args.rois.split(",") if r.strip()]

    missing = []
    for r in rois:
        for m in ("mag", "delay"):
            ok = ((pooled["roi"] == r) & (pooled["metric"] == m)).any()
            if not ok:
                missing.append((r, m))
    if missing:
        raise ValueError(
            "pooled_empirical_distributions.csv is missing these (roi,metric) pairs:\n"
            + "\n".join([f"  {r},{m}" for r, m in missing])
        )

    cfg = {}
    for r in rois:
        cfg[r] = {
            "n_vox": compute_nvox(qc, r, args.mode),

            "pooled_csv": str(pooled_path),

            "roi_value": r,
            "mag_metric": "mag",
            "delay_metric": "delay",
            "value_col": "value",
        }

    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(cfg, indent=2))
    print(f"Wrote: {out_path.resolve()}\n")

    print("Chosen n_vox (from your QC counts):")
    for r in rois:
        print(f"  {r:6s}  n_vox={cfg[r]['n_vox']}   (mode={args.mode})")

    print("\nOne-line JSON for --roi_config_json:")
    print(json.dumps(cfg, separators=(",", ":")))


if __name__ == "__main__":
    main()
