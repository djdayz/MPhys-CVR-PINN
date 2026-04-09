
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib

WM_LABELS = {2, 41}

SUBCORT_GM_LABELS = {
    10, 49,
    11, 50,
    12, 51,
    13, 52,
    17, 53,
    18, 54,
    26, 58,
    28, 60,
}

VCSF_LABELS = {4, 43, 5, 44, 14, 15}


def load_labelmap(p):
    img = nib.load(str(p))
    data = np.rint(img.get_fdata()).astype(np.int32)
    return img, data


def save_mask(mask, ref_img, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = mask.astype(np.uint8)
    out = nib.Nifti1Image(m, ref_img.affine, ref_img.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, str(out_path))


def process_subject_mri(mri_dir):
    aseg_p = mri_dir / "aseg.auto_noCCseg.mgz"
    aparc_p = mri_dir / "aparc.DKTatlas+aseg.deep.mgz"

    if not aseg_p.exists():
        raise FileNotFoundError(f"Missing: {aseg_p}")
    if not aparc_p.exists():
        raise FileNotFoundError(f"Missing: {aparc_p}")

    aseg_img, aseg = load_labelmap(aseg_p)
    aparc_img, aparc = load_labelmap(aparc_p)

    wm = np.isin(aseg, list(WM_LABELS))
    subcort_gm = np.isin(aseg, list(SUBCORT_GM_LABELS))
    vcsf = np.isin(aseg, list(VCSF_LABELS))

    cortical_gm = (aparc >= 1000) & (aparc < 3000)

    out_dir = mri_dir / "masks"
    save_mask(wm, aseg_img, out_dir / "wm_mask.nii.gz")
    save_mask(subcort_gm, aseg_img, out_dir / "subcort_gm_mask.nii.gz")
    save_mask(vcsf, aseg_img, out_dir / "vcsf_mask.nii.gz")
    save_mask(cortical_gm, aparc_img, out_dir / "cortical_gm_mask.nii.gz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids", required=True, type=Path, help="Path to bids_dir")
    args = ap.parse_args()

    bids = args.bids.expanduser().resolve()
    if not bids.exists():
        raise SystemExit(f"BIDS dir not found: {bids}")

    mri_dirs = sorted(bids.glob("sub-*/fastsurfer/mri"))

    if not mri_dirs:
        raise SystemExit(f"No FastSurfer mri dirs found under {bids} (expected sub-*/fastsurfer/mri)")

    print(f"Found {len(mri_dirs)} subjects with FastSurfer outputs.")

    ok = fail = 0
    for mri_dir in mri_dirs:
        sub = mri_dir.parts[-3]
        try:
            process_subject_mri(mri_dir)
            print(f"[OK]   {sub} -> {mri_dir / 'masks'}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {sub}: {e}")
            fail += 1

    print(f"Done. OK={ok}, FAIL={fail}")


if __name__ == "__main__":
    main()