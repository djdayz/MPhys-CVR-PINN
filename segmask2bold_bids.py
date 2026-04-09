
import os
import re
import sys
import glob
import shlex
import shutil
import subprocess
from pathlib import Path
import argparse
import nibabel as nib
import numpy as np


def run(cmd, check=True):
    print(">>", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=check)


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)


def first_existing(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def all_existing(patterns):
    out = []
    for pat in patterns:
        out.extend(glob.glob(pat))
    return sorted(list(dict.fromkeys(out)))


def fslorient_copy_qform2sform(img):
    run(["fslorient", "-copyqform2sform", str(img)])


def pick_best_mat(mat_candidates):

    if not mat_candidates:
        return None
    scored: List[Tuple[int, str]] = []
    for m in mat_candidates:
        name = os.path.basename(m).lower()
        score = 0
        if "t1" in name or "t1w" in name:
            score += 2
        if "bold" in name or "func" in name:
            score += 2
        if "to" in name:
            score += 1
        scored.append((score, m))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][1]


def mri_convert_to_nii(mgz_path, out_path):
    ensure_dir(out_path.parent)

    img = nib.load(str(mgz_path))
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine

    nii = nib.Nifti1Image(data, affine)
    hdr = nii.header
    hdr.set_qform(affine, code=1)
    hdr.set_sform(affine, code=1)

    nib.save(nii, str(out_path))


def maybe_gunzip_nii(path):
    return path


def sanitize_stem(p):
    name = p.name
    name = re.sub(r"\.nii(\.gz)?$", "", name)
    name = re.sub(r"\.mgz$", "", name)
    return name


def process_subject_session(bids_dir, sub_id, ses_id, overwrite=False):
    sub_dir = bids_dir / sub_id
    ses_dir = sub_dir / ses_id

    fastsurfer_mri = sub_dir / "fastsurfer" / "mri"
    aseg_mgz = fastsurfer_mri / "aseg.auto_noCCseg.mgz"
    masks_dir = fastsurfer_mri / "masks"

    if not aseg_mgz.exists():
        print(f"[SKIP] {sub_id}: missing {aseg_mgz}")
        return
    if not masks_dir.exists():
        print(f"[SKIP] {sub_id}: missing {masks_dir}")
        return

    t1w = first_existing([
        str(sub_dir / "anat" / f"{sub_id}_*T1w.nii.gz"),
        str(sub_dir / "anat" / f"{sub_id}_*T1w.nii"),
        str(sub_dir / "anat" / "*T1w.nii.gz"),
        str(sub_dir / "anat" / "*T1w.nii"),
    ])
    if t1w is None:
        print(f"[SKIP] {sub_id}: could not find raw T1w under {sub_dir/'anat'}")
        return
    t1w = str(Path(t1w))

    pre_dir = ses_dir / "pre"
    mean_bold = first_existing([
        str(pre_dir / "*mean*bold*.nii.gz"),
        str(pre_dir / "*mean*bold*.nii"),
        str(pre_dir / "*mean*.nii.gz"),
        str(pre_dir / "*boldmcf_mean_reg*.nii"),
    ])
    if mean_bold is None:
        print(f"[SKIP] {sub_id}/{ses_id}: could not find mean BOLD under {pre_dir}")
        return
    mean_bold = str(Path(mean_bold))

    t1_to_bold_dir = bids_dir / "t1_to_bold" / sub_id / ses_id / "anat"
    mat_candidates = sorted(glob.glob(str(t1_to_bold_dir / "*.mat")))
    t1_to_bold_mat = pick_best_mat(mat_candidates)
    if t1_to_bold_mat is None:
        print(f"[SKIP] {sub_id}/{ses_id}: could not find any .mat under {t1_to_bold_dir}")
        return

    outdir = ses_dir / "roi2bold"
    ensure_dir(outdir)

    roi_files = all_existing([
        str(masks_dir / "*.mgz"),
        str(masks_dir / "*.nii.gz"),
        str(masks_dir / "*.nii"),
    ])
    if not roi_files:
        print(f"[SKIP] {sub_id}: no ROI masks found in {masks_dir}")
        return

    for roi in roi_files:
        roi_path = Path(roi)
        stem = sanitize_stem(roi_path)

        if roi_path.suffix == ".mgz":
            roi_nii = outdir / f"{stem}.nii"
            if overwrite or not roi_nii.exists():
                mri_convert_to_nii(roi_path, roi_nii)
        else:
            roi_nii = Path(roi)
            roi_copy = outdir / f"{stem}{''.join(roi_nii.suffixes)}"
            if (overwrite or not roi_copy.exists()) and (roi_nii.resolve() != roi_copy.resolve()):
                shutil.copyfile(roi_nii, roi_copy)
            roi_nii = roi_copy

        roi_in_t1 = outdir / f"{stem}_in_T1w.nii"
        if overwrite or not roi_in_t1.exists():
            run([
                "flirt",
                "-in", str(roi_nii),
                "-ref", t1w,
                "-out", str(roi_in_t1),
                "-applyxfm",
                "-usesqform",
                "-interp", "nearestneighbour",
            ])

        if overwrite or True:
            run(["fslcpgeom", t1w, str(roi_in_t1)])
            fslorient_copy_qform2sform(roi_in_t1)

        roi_in_bold = outdir / f"{stem}_in_BOLD.nii"
        if overwrite or not roi_in_bold.exists():
            run([
                "flirt",
                "-in", str(roi_in_t1),
                "-ref", mean_bold,
                "-out", str(roi_in_bold),
                "-applyxfm",
                "-init", str(t1_to_bold_mat),
                "-interp", "nearestneighbour",
            ])

        roi_in_bold_bin = outdir / f"{stem}_in_BOLD_bin.nii"
        if overwrite or not roi_in_bold_bin.exists():
            run([
                "fslmaths", str(roi_in_bold),
                "-thr", "0.5", "-bin",
                str(roi_in_bold_bin),
            ])

    print(f"[DONE] {sub_id}/{ses_id} -> {outdir}")


def find_subjects(bids_dir):
    return sorted([p.name for p in bids_dir.glob("sub-*") if p.is_dir()])


def find_sessions(sub_dir):
    ses = sorted([p.name for p in sub_dir.glob("ses-*") if p.is_dir()])
    return ses


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--bids")
    parser.add_argument("--sub", default=None)
    parser.add_argument("--ses", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    bids_dir = Path(args.bids).resolve()

    if not bids_dir.exists():
        raise FileNotFoundError(bids_dir)

    if args.sub:
        subs = [args.sub]
    else:
        subs = find_subjects(bids_dir)

    for sub_id in subs:
        sub_dir = bids_dir / sub_id

        if args.ses:
            sessions = [args.ses]
        else:
            sessions = find_sessions(sub_dir)

        for ses_id in sessions:
            process_subject_session(bids_dir, sub_id, ses_id, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
