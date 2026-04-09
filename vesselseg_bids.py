import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None


def load_nii(path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return img, data


def save_mask_like(ref_img, mask_bool, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_data = mask_bool.astype(np.uint8)
    out_img = nib.Nifti1Image(out_data, ref_img.affine, ref_img.header)
    out_img.set_data_dtype(np.uint8)

    nib.save(out_img, str(out_path))


def find_first_existing(folder, names):

    for name in names:
        p = folder / name
        if p.exists():
            return p
    return None


def with_nii_variants(fname):

    p = Path(fname)
    if p.suffix == ".gz" and p.name.endswith(".nii.gz"):
        return [p.name, p.with_suffix("").name]
    if p.suffix == ".nii":
        return [p.name, p.name + ".gz"]
    return [fname, fname + ".nii.gz", fname + ".nii"]


def clean_mask(mask, min_cluster_size=0, do_closing=False):

    if ndi is None:
        return mask

    out = mask.copy()

    if do_closing:
        struct = ndi.generate_binary_structure(3, 2)
        out = ndi.binary_closing(out, structure=struct, iterations=1)

    if min_cluster_size and min_cluster_size > 0:
        struct = ndi.generate_binary_structure(3, 2)
        labeled, n = ndi.label(out, structure=struct)
        if n > 0:
            counts = np.bincount(labeled.ravel())
            keep = np.zeros_like(counts, dtype=bool)
            keep[counts >= min_cluster_size] = True
            keep[0] = False
            out = keep[labeled]

    return out


def segment_one_session(ses_dir, cvr_dirname, roi2bold_dirname, tcnr_name, cvrmag_name, vcsf_mask_name, out_name, cvr_abs_thresh, overwrite, closing, min_cluster_size):
    cvr_dir = ses_dir / cvr_dirname
    roi2bold_dir = ses_dir / roi2bold_dirname

    tcnr_path = find_first_existing(cvr_dir, with_nii_variants(tcnr_name))
    cvr_path = find_first_existing(cvr_dir, with_nii_variants(cvrmag_name))
    vcsf_path = find_first_existing(roi2bold_dir, with_nii_variants(vcsf_mask_name))

    if tcnr_path is None or cvr_path is None or vcsf_path is None:
        return False, (
            f"Missing file(s): "
            f"tcnr={tcnr_path}, cvr={cvr_path}, vcsf={vcsf_path} "
            f"(looked in {cvr_dir} and {roi2bold_dir})"
        )

    out_path = roi2bold_dir / out_name
    if out_path.exists() and not overwrite:
        return True, f"Exists (skip): {out_path}"

    tcnr_img, t = load_nii(tcnr_path)
    cvr_img, c = load_nii(cvr_path)
    _, vcsf = load_nii(vcsf_path)

    if t.shape != c.shape or t.shape != vcsf.shape:
        return False, f"Shape mismatch: tcnr{t.shape} cvr{c.shape} vcsf{vcsf.shape}"

    finite = np.isfinite(t) & np.isfinite(c) & np.isfinite(vcsf)

    vessel = np.zeros(t.shape, dtype=bool)
    cond1 = np.abs(c) > float(cvr_abs_thresh)
    cond2 = (t < -0.5) & (c < -0.3)
    cond3 = np.abs(t) > 4.5
    vessel[finite] = (cond1 | cond2 | cond3)[finite]

    vcsf_bin = (vcsf > 0.5) & finite
    vessel[vcsf_bin] = False

    vessel = clean_mask(vessel, min_cluster_size=min_cluster_size, do_closing=closing)

    save_mask_like(tcnr_img, vessel, out_path)

    return True, f"Wrote: {out_path} (voxels={int(vessel.sum())})"


def iter_sub_ses(bids_dir, sub=None, ses=None):
    sub_dirs = [bids_dir / sub] if sub else sorted(bids_dir.glob("sub-*"))
    for sub_dir in sub_dirs:
        if not sub_dir.is_dir():
            continue
        ses_dirs = [sub_dir / ses] if ses else sorted(sub_dir.glob("ses-*"))
        for ses_dir in ses_dirs:
            if ses_dir.is_dir():
                yield sub_dir, ses_dir


def main():
    ap = argparse.ArgumentParser(
        description="Blood vessel segmentation using CVR magnitude and tCNR, then subtract vCSF."
    )
    ap.add_argument("--bids_dir", required=True, type=Path)

    ap.add_argument("--sub", default=None, help="e.g. sub-01 (optional)")
    ap.add_argument("--ses", default=None, help="e.g. ses-01 (optional)")

    ap.add_argument("--cvr_dirname", default="cvr", help="folder under ses-* containing tCNR and CVR maps")
    ap.add_argument("--roi2bold_dirname", default="roi2bold", help="folder under ses-* containing vCSF mask")

    ap.add_argument("--tcnr_name", default="tCNR.nii.gz", help="tCNR filename inside cvr_dirname")
    ap.add_argument("--cvrmag_name", default="CVR_mag.nii.gz", help="CVR magnitude filename inside cvr_dirname")
    ap.add_argument("--vcsf_mask_name", default="vcsf_mask_in_BOLD_bin.nii.gz", help="vCSF mask inside roi2bold_dirname")

    ap.add_argument("--out_name", default="vessel_mask_in_BOLD_bin.nii", help="output filename in roi2bold_dirname")

    ap.add_argument("--cvr_abs_thresh", type=float, default=0.9, help="threshold for |CVR_mag|")
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--closing", action="store_true", help="binary closing (needs scipy)")
    ap.add_argument("--min_cluster_size", type=int, default=5, help="remove connected components smaller than this (needs scipy)")

    args = ap.parse_args()

    if not args.bids_dir.exists():
        print(f"ERROR: bids_dir not found: {args.bids_dir}", file=sys.stderr)
        sys.exit(1)

    ok_all = True
    for sub_dir, ses_dir in iter_sub_ses(args.bids_dir, args.sub, args.ses):
        ok, msg = segment_one_session(
            ses_dir=ses_dir,
            cvr_dirname=args.cvr_dirname,
            roi2bold_dirname=args.roi2bold_dirname,
            tcnr_name=args.tcnr_name,
            cvrmag_name=args.cvrmag_name,
            vcsf_mask_name=args.vcsf_mask_name,
            out_name=args.out_name,
            cvr_abs_thresh=args.cvr_abs_thresh,
            overwrite=args.overwrite,
            closing=args.closing,
            min_cluster_size=args.min_cluster_size,
        )
        print(f"[{sub_dir.name}/{ses_dir.name}] {msg}")
        ok_all = ok_all and ok

    sys.exit(0 if ok_all else 2)


if __name__ == "__main__":
    main()
