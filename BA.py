
import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def strip_nii_suffix(path):
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def extract_case_id(path, regex):

    s = str(path)
    if regex is not None:
        m = re.search(regex, s)
        if m:
            return m.group(0)
    return path.parent.name


def build_case_map(paths, case_id_regex):
    case_map: Dict[str, Path] = {}
    for p in paths:
        cid = extract_case_id(p, case_id_regex)
        if cid in case_map:
            raise ValueError(f"Duplicate case id '{cid}' for:\n  {case_map[cid]}\n  {p}")
        case_map[cid] = p
    return case_map


def load_nifti_data(path):
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    return img, data


def same_grid(img1, img2):
    same_shape = img1.shape == img2.shape
    same_aff = np.allclose(img1.affine, img2.affine, atol=1e-4)
    return same_shape and same_aff


def bland_altman_stats(x, y):

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.shape != y.shape:
        raise ValueError(f"x and y must have same shape, got {x.shape} vs {y.shape}")

    diff = x - y
    meanv = 0.5 * (x + y)

    n = diff.size
    bias = float(np.mean(diff))

    if n > 1:
        sd = float(np.std(diff, ddof=1))
        loa_low = bias - 1.96 * sd
        loa_high = bias + 1.96 * sd
    else:
        sd = np.nan
        loa_low = np.nan
        loa_high = np.nan

    return {
        "n": int(n),
        "bias": bias,
        "sd_diff": sd,
        "loa_low": float(loa_low),
        "loa_high": float(loa_high),
        "mean_of_means": float(np.mean(meanv)),
    }


def default_mask(a, b):
    
    finite = np.isfinite(a) & np.isfinite(b)
    nonzero = (np.abs(a) > 0.009) | (np.abs(b) > 0.009)
    mask = finite & nonzero

    if not np.any(mask):
        mask = finite
    return mask


def sample_arrays(a, b, max_n, rng):
    if max_n is None or a.size <= max_n:
        return a, b
    idx = rng.choice(a.size, size=max_n, replace=False)
    return a[idx], b[idx]


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_bland_altman(meanv, diff, stats, out_path, title, y_label, point_alpha=0.25, point_size=8.0):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=160)

    ax.scatter(meanv, diff, s=point_size, alpha=point_alpha, edgecolors="none")

    ax.axhline(0.0, linestyle=":", linewidth=1.0, label="zero difference")
    ax.axhline(stats["bias"], linestyle="--", linewidth=1.5, label=f"bias = {stats['bias']:.4g}")
    ax.axhline(stats["loa_low"], linestyle="-.", linewidth=1.2, label=f"lower LoA = {stats['loa_low']:.4g}")
    ax.axhline(stats["loa_high"], linestyle="-.", linewidth=1.2, label=f"upper LoA = {stats['loa_high']:.4g}")

    ax.set_title(title)
    ax.set_xlabel("Mean of methods: (PINN + Supervised) / 2")
    ax.set_ylabel(y_label)
    ax.legend(frameon=False, fontsize=9)

    txt = (
        f"n = {stats['n']}\n"
        f"bias = {stats['bias']:.4g}\n"
        f"SD(diff) = {stats['sd_diff']:.4g}\n"
        f"95% LoA = [{stats['loa_low']:.4g}, {stats['loa_high']:.4g}]"
    )
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def analyze_metric(metric_name, pinn_case_map, sup_case_map, outdir, case_id_regex, mask_case_map=None, shared_mask_path=None, sample_voxels_per_case=20000, seed=0, diff_label=None):
    rng = np.random.default_rng(seed)

    common_cases = sorted(set(pinn_case_map).intersection(sup_case_map))
    if not common_cases:
        raise RuntimeError(f"No paired cases found for {metric_name}.")

    case_rows = []
    pooled_pinn = []
    pooled_sup = []

    shared_mask_data = None
    shared_mask_img = None
    if shared_mask_path is not None:
        shared_mask_img, shared_mask_data = load_nifti_data(shared_mask_path)
        shared_mask_data = shared_mask_data > 0

    for cid in common_cases:
        pinn_path = pinn_case_map[cid]
        sup_path = sup_case_map[cid]

        pinn_img, pinn_data = load_nifti_data(pinn_path)
        sup_img, sup_data = load_nifti_data(sup_path)

        if not same_grid(pinn_img, sup_img):
            raise ValueError(
                f"Grid mismatch for case {cid}\n"
                f"  PINN: {pinn_path}\n"
                f"  SUP : {sup_path}"
            )

        if shared_mask_data is not None:
            if not same_grid(pinn_img, shared_mask_img):
                raise ValueError(
                    f"Shared mask grid mismatch for case {cid}\n"
                    f"  map : {pinn_path}\n"
                    f"  mask: {shared_mask_path}"
                )
            mask = shared_mask_data.copy()

        elif mask_case_map is not None and cid in mask_case_map:
            mask_img, mask_data = load_nifti_data(mask_case_map[cid])
            if not same_grid(pinn_img, mask_img):
                raise ValueError(
                    f"Mask grid mismatch for case {cid}\n"
                    f"  map : {pinn_path}\n"
                    f"  mask: {mask_case_map[cid]}"
                )
            mask = mask_data > 0

        else:
            mask = default_mask(pinn_data, sup_data)

        valid = mask & np.isfinite(pinn_data) & np.isfinite(sup_data)
        pinn_vox = pinn_data[valid].astype(np.float64)
        sup_vox = sup_data[valid].astype(np.float64)

        if pinn_vox.size == 0:
            print(f"[WARN] case {cid}: no valid voxels after masking, skipping")
            continue

        pinn_mean = float(np.mean(pinn_vox))
        sup_mean = float(np.mean(sup_vox))
        pair_mean = 0.5 * (pinn_mean + sup_mean)
        diff_mean = pinn_mean - sup_mean

        voxel_diff = pinn_vox - sup_vox
        case_rows.append({
            "case_id": cid,
            "n_voxels": int(pinn_vox.size),
            "pinn_mean": pinn_mean,
            "supervised_mean": sup_mean,
            "pair_mean": pair_mean,
            "diff_mean_pinn_minus_supervised": diff_mean,
            "voxel_diff_mean": float(np.mean(voxel_diff)),
            "voxel_diff_sd": float(np.std(voxel_diff, ddof=1)) if voxel_diff.size > 1 else np.nan,
            "pinn_path": str(pinn_path),
            "supervised_path": str(sup_path),
        })

        pinn_sample, sup_sample = sample_arrays(
            pinn_vox, sup_vox, sample_voxels_per_case, rng
        )
        pooled_pinn.append(pinn_sample)
        pooled_sup.append(sup_sample)

    if not case_rows:
        raise RuntimeError(f"All cases were skipped for {metric_name}.")

    case_df = pd.DataFrame(case_rows).sort_values("case_id").reset_index(drop=True)
    outdir.mkdir(parents=True, exist_ok=True)
    case_csv = outdir / f"{metric_name}_per_case_summary.csv"
    case_df.to_csv(case_csv, index=False)

    case_x = case_df["pinn_mean"].to_numpy(dtype=np.float64)
    case_y = case_df["supervised_mean"].to_numpy(dtype=np.float64)

    case_stats = bland_altman_stats(case_x, case_y)
    save_json(case_stats, outdir / f"{metric_name}_case_mean_ba_stats.json")

    case_meanv = 0.5 * (case_x + case_y)
    case_diff = case_x - case_y

    plot_bland_altman(
        meanv=case_meanv,
        diff=case_diff,
        stats=case_stats,
        out_path=outdir / f"{metric_name}_case_mean_bland_altman.png",
        title=f"{metric_name.upper()} each case",
        y_label=diff_label or "Difference: PINN - Supervised",
        point_alpha=0.8,
        point_size=28,
    )

    pooled_x = np.concatenate(pooled_pinn, axis=0)
    pooled_y = np.concatenate(pooled_sup, axis=0)

    pooled_stats = bland_altman_stats(pooled_x, pooled_y)
    save_json(pooled_stats, outdir / f"{metric_name}_voxelwise_ba_stats.json")

    pooled_meanv = 0.5 * (pooled_x + pooled_y)
    pooled_diff = pooled_x - pooled_y

    plot_bland_altman(
        meanv=pooled_meanv,
        diff=pooled_diff,
        stats=pooled_stats,
        out_path=outdir / f"{metric_name}_voxelwise_bland_altman.png",
        title=f"{metric_name.upper()} voxelwise (pooled sampled voxels)",
        y_label=diff_label or "Difference: PINN - Supervised",
        point_alpha=0.18,
        point_size=6,
    )

    print(f"[OK] {metric_name}:")
    print(f"  cases analysed: {len(case_df)}")
    print(f"  case CSV      : {case_csv}")
    print(f"  case BA plot  : {outdir / f'{metric_name}_case_mean_bland_altman.png'}")
    print(f"  voxel BA plot : {outdir / f'{metric_name}_voxelwise_bland_altman.png'}")
    print(f"  case stats    : {outdir / f'{metric_name}_case_mean_ba_stats.json'}")
    print(f"  voxel stats   : {outdir / f'{metric_name}_voxelwise_ba_stats.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Bland–Altman comparison between PINN and supervised CVR maps."
    )

    parser.add_argument("--pinn-mag-glob", type=str, default="/Users/mac/workspace/pinn_tcnr5.0_40/mag/*.nii.gz",
                        help="Glob for PINN CVR magnitude maps.")
    parser.add_argument("--sup-mag-glob", type=str, default="/Users/mac/workspace/sup_5.0/test_inference_maps/mag/*.nii.gz",
                        help="Glob for supervised CVR magnitude maps.")

    parser.add_argument("--pinn-delay-glob", type=str, default="/Users/mac/workspace/pinn_tcnr5.0_40/delay/*.nii.gz",
                        help="Glob for PINN CVR delay maps.")
    parser.add_argument("--sup-delay-glob", type=str, default="/Users/mac/workspace/sup_5.0/test_inference_maps/delay/*.nii.gz",
                        help="Glob for supervised CVR delay maps.")

    parser.add_argument("--mask-glob", type=str, default=None,
                        help="Optional glob for one mask per case.")
    parser.add_argument("--mask-file", type=str, default=None,
                        help="Optional single mask applied to all cases.")

    parser.add_argument("--case-id-regex", type=str, default=r"rep\d+",
                        help="Regex used to pair corresponding cases. Default: rep\\d+")
    parser.add_argument("--sample-voxels-per-case", type=int, default=20000,
                        help="Max voxels per case for pooled voxelwise BA plot.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="/Users/mac/workspace/ba_analysis_5.0")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mask_case_map = None
    if args.mask_glob:
        mask_paths = sorted(Path().glob(args.mask_glob)) if not any(ch in args.mask_glob for ch in "*?[]") else sorted(Path(p) for p in __import__("glob").glob(args.mask_glob))
        mask_case_map = build_case_map(mask_paths, args.case_id_regex)
    shared_mask_path = Path(args.mask_file) if args.mask_file else None

    any_metric = False

    if args.pinn_mag_glob and args.sup_mag_glob:
        pinn_mag_paths = sorted(Path(p) for p in __import__("glob").glob(args.pinn_mag_glob))
        sup_mag_paths = sorted(Path(p) for p in __import__("glob").glob(args.sup_mag_glob))

        if not pinn_mag_paths:
            raise RuntimeError(f"No PINN magnitude files matched: {args.pinn_mag_glob}")
        if not sup_mag_paths:
            raise RuntimeError(f"No supervised magnitude files matched: {args.sup_mag_glob}")

        pinn_mag_map = build_case_map(pinn_mag_paths, args.case_id_regex)
        sup_mag_map = build_case_map(sup_mag_paths, args.case_id_regex)

        analyze_metric(
            metric_name="CVR magnitude",
            pinn_case_map=pinn_mag_map,
            sup_case_map=sup_mag_map,
            outdir=outdir,
            case_id_regex=args.case_id_regex,
            mask_case_map=mask_case_map,
            shared_mask_path=shared_mask_path,
            sample_voxels_per_case=args.sample_voxels_per_case,
            seed=args.seed,
            diff_label="Difference in CVR magnitude: PINN - Supervised (%/mmHg)",
        )
        any_metric = True

    if args.pinn_delay_glob and args.sup_delay_glob:
        pinn_delay_paths = sorted(Path(p) for p in __import__("glob").glob(args.pinn_delay_glob))
        sup_delay_paths = sorted(Path(p) for p in __import__("glob").glob(args.sup_delay_glob))

        if not pinn_delay_paths:
            raise RuntimeError(f"No PINN delay files matched: {args.pinn_delay_glob}")
        if not sup_delay_paths:
            raise RuntimeError(f"No supervised delay files matched: {args.sup_delay_glob}")

        pinn_delay_map = build_case_map(pinn_delay_paths, args.case_id_regex)
        sup_delay_map = build_case_map(sup_delay_paths, args.case_id_regex)

        analyze_metric(
            metric_name="CVR delay",
            pinn_case_map=pinn_delay_map,
            sup_case_map=sup_delay_map,
            outdir=outdir,
            case_id_regex=args.case_id_regex,
            mask_case_map=mask_case_map,
            shared_mask_path=shared_mask_path,
            sample_voxels_per_case=args.sample_voxels_per_case,
            seed=args.seed + 1,
            diff_label="Difference in CVR delay: PINN - Supervised (s)",
        )
        any_metric = True

    if not any_metric:
        raise RuntimeError("You must provide at least one complete pair of globs: magnitude and/or delay.")

    print("\nDone.")


if __name__ == "__main__":
    main()
