
import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


def load_nifti_data(path, volume_index=None):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)

    if data.ndim == 4:
        if volume_index is None:
            raise ValueError(
                f"{path} is 4D with shape {data.shape}. "
                "Please pass --volume-index to choose one volume."
            )
        if not (0 <= volume_index < data.shape[3]):
            raise ValueError(f"--volume-index {volume_index} out of range for shape {data.shape}")
        data = data[..., volume_index]

    return img, np.asarray(data, dtype=np.float32)


def save_nifti_like(ref_img, data, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(np.float32)
    out = nib.Nifti1Image(data.astype(np.float32), ref_img.affine, hdr)
    nib.save(out, str(out_path))


def ensure_same_shape(*arrays):
    shapes = [a.shape for a in arrays]
    if len(set(shapes)) != 1:
        raise ValueError(f"Input shapes do not match: {shapes}")


def build_valid_mask(gt, pinn, sup, mask=None):
    valid = np.isfinite(gt) & np.isfinite(pinn) & np.isfinite(sup)
    if mask is not None:
        valid &= mask > 0
    return valid


def masked_pcc(gt, pred, valid_mask):
    x = gt[valid_mask]
    y = pred[valid_mask]

    if x.size < 2:
        return float("nan")

    sx = np.std(x)
    sy = np.std(y)
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def bounding_box_from_mask(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))


def masked_ssim_bbox(gt, pred, valid_mask, win_size=None):

    if not HAS_SKIMAGE:
        return float("nan")

    bbox = bounding_box_from_mask(valid_mask)
    if bbox is None:
        return float("nan")

    gt_crop = gt[bbox].copy()
    pred_crop = pred[bbox].copy()
    mask_crop = valid_mask[bbox]

    gt_crop[~mask_crop] = 0.0
    pred_crop[~mask_crop] = 0.0

    vals = np.concatenate([gt_crop[mask_crop], pred_crop[mask_crop]])
    if vals.size == 0:
        return float("nan")

    data_range = float(vals.max() - vals.min())
    if data_range <= 0:
        data_range = 1.0

    kwargs = dict(data_range=data_range)
    if win_size is not None:
        kwargs["win_size"] = win_size

    try:
        return float(ssim(gt_crop, pred_crop, **kwargs))
    except Exception:
        return float("nan")


def compute_pre_map(gt, pred, valid_mask, eps=1e-6, absolute=True):

    out = np.zeros_like(gt, dtype=np.float32)

    denom = np.maximum(np.abs(gt), eps)
    valid = valid_mask & np.isfinite(gt) & np.isfinite(pred)

    if absolute:
        out[valid] = 100.0 * np.abs(pred[valid] - gt[valid]) / denom[valid]
    else:
        out[valid] = 100.0 * (pred[valid] - gt[valid]) / denom[valid]

    return out, valid


def masked_mean(arr, mask):
    vals = arr[mask]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def masked_median(arr, mask):
    vals = arr[mask]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def compare_one_prediction(name, gt, pred, valid_mask, ref_img, outdir, pre_eps, signed_pre, ssim_win):
    pcc_val = masked_pcc(gt, pred, valid_mask)
    ssim_val = masked_ssim_bbox(gt, pred, valid_mask, win_size=ssim_win)

    pre_map, pre_valid = compute_pre_map(
        gt=gt,
        pred=pred,
        valid_mask=valid_mask,
        eps=pre_eps,
        absolute=not signed_pre,
    )

    save_nifti_like(ref_img, pre_map, outdir / f"{name}_pre_map.nii.gz")
    save_nifti_like(ref_img, pre_valid.astype(np.float32), outdir / f"{name}_pre_valid_mask.nii.gz")

    result = {
        "comparison": name,
        "pcc": pcc_val,
        "ssim": ssim_val,
        "mean_pre_percent": masked_mean(pre_map, pre_valid),
        "median_pre_percent": masked_median(pre_map, pre_valid),
        "num_valid_voxels": int(np.count_nonzero(pre_valid)),
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute scalar SSIM/PCC and PRE maps for PINN vs GT and supervised vs GT."
    )
    parser.add_argument("--gt", type=Path, required=True, help="Ground truth NIfTI map")
    parser.add_argument("--pinn", type=Path, required=True, help="PINN prediction NIfTI map")
    parser.add_argument("--sup", type=Path, required=True, help="Supervised prediction NIfTI map")
    parser.add_argument("--mask", type=Path, default=None, help="Optional mask NIfTI")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory")
    parser.add_argument("--volume-index", type=int, default=None, help="If maps are 4D, choose which volume to use")
    parser.add_argument("--pre-eps", type=float, default=1e-6, help="Small epsilon for PRE denominator")
    parser.add_argument("--signed-pre", action="store_true", help="Use signed PRE instead of absolute PRE")
    parser.add_argument(
        "--ssim-win",
        type=int,
        default=None,
        help="Optional SSIM window size (must be odd). Leave unset to use skimage default."
    )
    args = parser.parse_args()

    if not HAS_SKIMAGE:
        raise ImportError(
            "scikit-image is required for SSIM. Install it with:\n"
            "pip install scikit-image"
        )

    args.outdir.mkdir(parents=True, exist_ok=True)

    gt_img, gt = load_nifti_data(args.gt, args.volume_index)
    _, pinn = load_nifti_data(args.pinn, args.volume_index)
    _, sup = load_nifti_data(args.sup, args.volume_index)

    ensure_same_shape(gt, pinn, sup)

    mask_data = None
    if args.mask is not None:
        _, mask_data = load_nifti_data(args.mask, args.volume_index)
        if mask_data.shape != gt.shape:
            raise ValueError(f"Mask shape {mask_data.shape} does not match map shape {gt.shape}")

    valid_mask = build_valid_mask(gt, pinn, sup, mask_data)
    if np.count_nonzero(valid_mask) == 0:
        raise ValueError("No valid voxels found after masking.")

    save_nifti_like(gt_img, valid_mask.astype(np.float32), args.outdir / "comparison_mask.nii.gz")

    pinn_result = compare_one_prediction(
        name="pinn_vs_gt",
        gt=gt,
        pred=pinn,
        valid_mask=valid_mask,
        ref_img=gt_img,
        outdir=args.outdir,
        pre_eps=args.pre_eps,
        signed_pre=args.signed_pre,
        ssim_win=args.ssim_win,
    )

    sup_result = compare_one_prediction(
        name="sup_vs_gt",
        gt=gt,
        pred=sup,
        valid_mask=valid_mask,
        ref_img=gt_img,
        outdir=args.outdir,
        pre_eps=args.pre_eps,
        signed_pre=args.signed_pre,
        ssim_win=args.ssim_win,
    )

    summary = {
        "gt": str(args.gt),
        "pinn": str(args.pinn),
        "sup": str(args.sup),
        "mask": str(args.mask) if args.mask is not None else None,
        "volume_index": args.volume_index,
        "pre_eps": args.pre_eps,
        "pre_type": "signed" if args.signed_pre else "absolute",
        "results": {
            "pinn_vs_gt": pinn_result,
            "sup_vs_gt": sup_result,
        },
    }

    with open(args.outdir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    csv_rows = [
        pinn_result,
        sup_result,
    ]
    with open(args.outdir / "metrics_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "comparison",
                "pcc",
                "ssim",
                "mean_pre_percent",
                "median_pre_percent",
                "num_valid_voxels",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()