
import argparse
import shutil
import subprocess
from pathlib import Path


ROI_MASKS = {
    "cgm": "cortical_gm_mask_in_BOLD_bin.nii",
    "sgm": "subcort_gm_mask_in_BOLD_bin.nii",
    "wm":  "wm_mask_in_BOLD_bin.nii",
    "vcsf": "vcsf_mask_in_BOLD_bin.nii",
    "vessel": "vessel_mask_in_BOLD_bin.nii",
}


def run(cmd):

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        msg = (
            f"Command failed:\n  {' '.join(cmd)}\n\n"
            f"STDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"
        )
        raise RuntimeError(msg) from e


def resolve_existing_mask(path_base):

    if path_base.exists():
        return path_base
    gz = path_base.with_suffix(path_base.suffix + ".gz")
    if gz.exists():
        return gz
    if path_base.name.endswith(".nii.gz"):
        nii = Path(str(path_base).replace(".nii.gz", ".nii"))
        if nii.exists():
            return nii
    return None


def out_name_with_suffix(in_path, suffix):
    name = in_path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        stem = name
    return in_path.parent / f"{stem}{suffix}.nii.gz"


def fsl_ero_n(in_path, out_path, iters, work_dir):

    if iters <= 0:
        run(["fslmaths", str(in_path), "-bin", str(out_path)])
        return

    tmp = work_dir / f".__tmp_ero_{out_path.stem}.nii.gz"
    run(["fslmaths", str(in_path), "-ero", str(tmp)])

    for _ in range(iters - 1):
        run(["fslmaths", str(tmp), "-ero", str(tmp)])

    run(["fslmaths", str(tmp), "-bin", str(out_path)])

    if tmp.exists():
        tmp.unlink(missing_ok=True)


def fsl_subtract_and_bin(a_path, b_path, out_path):

    run(["fslmaths", str(a_path), "-sub", str(b_path), "-thr", "0", "-bin", str(out_path)])


def parse_csv_list(s):
    if s is None:
        return None
    items = [x.strip() for x in s.split(",") if x.strip()]
    return set(items) if items else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids_dir", required=True, type=str)

    ap.add_argument("--subs", default=None, type=str,
                    help="Comma-separated subjects to include (e.g. sub-01,sub-03). If set, only these are processed.")
    ap.add_argument("--exclude_subs", default=None, type=str,
                    help="Comma-separated subjects to skip (e.g. sub-02,sub-04).")
    ap.add_argument("--sessions", default=None, type=str,
                    help="Comma-separated sessions to include (e.g. ses-01,ses-02). If set, only these are processed.")
    ap.add_argument("--exclude_sessions", default=None, type=str,
                    help="Comma-separated sessions to skip (e.g. ses-02).")

    ap.add_argument("--iters", type=int, default=1, help="Number of -ero iterations.")
    ap.add_argument("--suffix", type=str, default=None, help="Output suffix (default: _ero{iters}).")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--include", type=str, default="cgm,sgm,wm,vcsf",
                    help="Comma-separated ROIs to erode (default: cgm,sgm,wm,vcsf).")
    ap.add_argument("--subtract_vessel", action="store_true",
                    help="Subtract vessel mask (if exists) from each eroded tissue mask.")
    ap.add_argument("--subtract_vcsf_from_gm_wm", action="store_true",
                    help="Subtract vcsf mask (if exists) from cgm/sgm/wm after erosion.")
    args = ap.parse_args()

    if shutil.which("fslmaths") is None:
        raise RuntimeError(
            "Could not find `fslmaths` on PATH. Activate FSL environment first "
            "(e.g., source $FSLDIR/etc/fslconf/fsl.sh)."
        )

    bids_dir = Path(args.bids_dir)

    include_subs = parse_csv_list(args.subs)
    exclude_subs = parse_csv_list(args.exclude_subs) or set()
    include_ses = parse_csv_list(args.sessions)
    exclude_ses = parse_csv_list(args.exclude_sessions) or set()

    rois_to_erode = [r.strip() for r in args.include.split(",") if r.strip()]
    suffix = args.suffix or f"_ero{args.iters}"

    n_written = 0
    n_skipped_sub = 0
    n_skipped_ses = 0

    for sub_dir in sorted(bids_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        sub = sub_dir.name

        if include_subs is not None and sub not in include_subs:
            n_skipped_sub += 1
            continue
        if sub in exclude_subs:
            n_skipped_sub += 1
            continue

        for ses_dir in sorted(sub_dir.glob("ses-*")):
            if not ses_dir.is_dir():
                continue
            ses = ses_dir.name

            if include_ses is not None and ses not in include_ses:
                n_skipped_ses += 1
                continue
            if ses in exclude_ses:
                n_skipped_ses += 1
                continue

            roi2bold_dir = ses_dir / "roi2bold"
            if not roi2bold_dir.exists():
                continue

            vessel_in = resolve_existing_mask(roi2bold_dir / ROI_MASKS["vessel"])
            vcsf_in = resolve_existing_mask(roi2bold_dir / ROI_MASKS["vcsf"])

            for roi in rois_to_erode:
                if roi not in ROI_MASKS:
                    print(f"[WARN] Unknown ROI '{roi}' (known: {list(ROI_MASKS)})")
                    continue

                in_path = resolve_existing_mask(roi2bold_dir / ROI_MASKS[roi])
                if in_path is None:
                    continue

                out_path = out_name_with_suffix(in_path, suffix)
                if out_path.exists() and not args.overwrite:
                    continue

                fsl_ero_n(in_path, out_path, iters=args.iters, work_dir=roi2bold_dir)

                n_written += 1
                print(f"[{sub}/{ses}] wrote {out_path.name}")

    print(
        f"\nDone. Wrote {n_written} masks. "
        f"Skipped subs: {n_skipped_sub} (selection/exclusion). "
        f"Skipped ses: {n_skipped_ses} (selection/exclusion)."
    )


if __name__ == "__main__":
    main()
