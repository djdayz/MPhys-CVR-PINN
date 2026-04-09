

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib


def load_labels_as_int(path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)

    lab = np.rint(data).astype(np.int32)

    return lab, img


def make_mask(label_vol, labels):
    if len(labels) == 0:
        raise ValueError("labels set is empty")
    labels_arr = np.fromiter(labels, dtype=np.int32)
    mask = np.isin(label_vol, labels_arr).astype(np.uint8)
    return mask


def save_mask(mask, ref_img, out_path):
    out_img = nib.Nifti1Image(mask, ref_img.affine, ref_img.header)
    out_img.set_data_dtype(np.uint8)
    nib.save(out_img, str(out_path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels_nifti", required=True, type=Path, help="MIDA label NIfTI (integer label map).")
    p.add_argument("--out_dir", required=True, type=Path, help="Output directory for masks.")
    p.add_argument(
        "--include_cerebellum",
        action="store_true",
        help="Include cerebellar WM (9) in WM and cerebellar GM (2) in cGM.",
    )
    p.add_argument(
        "--include_general_csf",
        action="store_true",
        help="Include general CSF label (32) in vCSF mask (makes it 'all CSF' rather than ventricles-only).",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    label_vol, ref_img = load_labels_as_int(args.labels_nifti)


    wm_labels = {9, 12}
    cgm_labels = {2, 10}
    sgm_labels = {4, 5, 7, 8, 16, 17, 20, 21, 99, 116}
    vcsf_labels = {6}
    vessel_labels = {24, 25}

    if args.include_cerebellum:
        wm_labels.add(9)
        cgm_labels.add(2)

    if args.include_general_csf:
        vcsf_labels.add(32)

    outputs = {
        "wm_mask.nii": wm_labels,
        "cgm_mask.nii": cgm_labels,
        "sgm_mask.nii": sgm_labels,
        "vcsf_mask.nii": vcsf_labels,
        "vessel_mask.nii": vessel_labels
    }

    for fname, labs in outputs.items():
        mask = make_mask(label_vol, labs)
        out_path = args.out_dir / fname
        save_mask(mask, ref_img, out_path)
        nvox = int(mask.sum())
        print(f"[OK] {fname}: labels={sorted(labs)}  voxels={nvox}")

    print(f"\nDone. Masks saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
