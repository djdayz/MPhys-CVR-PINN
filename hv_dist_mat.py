import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.io import savemat

EXCLUDE: set(Tuple[str, str]) = {
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
    "wm": "wm_mask_in_BOLD_bin.nii",
    "vcsf": "vcsf_mask_in_BOLD_bin.nii",
    "vessel": "vessel_mask_in_BOLD_bin.nii",
}

def load_nii(path):
    
    img = nib.load(path.as_posix())
    data = img.get_fdata(dtype=np.float32)
    return data

def collect_session_voxels(cvr_mag_path, cvr_delay_path, roi_path, mask_thr=0.5):
    
    cvr_mag = load_nii(cvr_mag_path)
    cvr_delay = load_nii(cvr_delay_path)

    if cvr_mag.shape != cvr_delay.shape or cvr_mag.shape != load_nii(roi_path).shape:
        raise ValueError(
            f"Shape mismatch: cvr_mag {cvr_mag.shape}, cvr_delay {cvr_delay.shape}, roi_mask {load_nii(roi_path).shape}"
        )

    roi_mask = load_nii(roi_path) > mask_thr
    
    voxels_cvr_mag = cvr_mag[roi_mask].reshape(-1)
    voxels_cvr_delay = cvr_delay[roi_mask].reshape(-1)

    good = np.isfinite(voxels_cvr_mag) & np.isfinite(voxels_cvr_delay)
    voxels_cvr_mag = voxels_cvr_mag[good]
    voxels_cvr_delay = voxels_cvr_delay[good]

    return voxels_cvr_mag.astype(np.float32), voxels_cvr_delay.astype(np.float32)

def main():
    ap = argparse.ArgumentParser(description="Extract CVR voxels from fMRI data")
    ap.add_argument("--bids_dir", type=Path, help="Path to the BIDS dataset directory")
    ap.add_argument("--out_dir", type=Path, help="Directory to save the output .mat file")
    ap.add_argument("--mask_thr", type=float, default=0.5, help="Threshold for ROI mask (default: 0.5)")
    ap.add_argument("--verbose", action="store_true", help="Print progress information")
    args = ap.parse_args()

    bids_dir: Path = args.bids_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled: Dict[str, Dict[str, List[float]]] = {roi: {"cvr_mag": [], "cvr_delay": []} for roi in ROI_FILES.keys()}

    n_used = 0
    n_skipped_excl = 0
    n_skipped_missing = 0

    for sub_dir in sorted(bids_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        sub = sub_dir.name

        for ses_dir in sorted(sub_dir.glob("ses-*")):
            if not ses_dir.is_dir():
                continue   
            ses = ses_dir.name

            if (sub, ses) in EXCLUDE:
                n_skipped_excl += 1
                if args.verbose:
                    print(f"Skipping {sub} {ses} (excluded)")
                continue

            cvr_dir = ses_dir / "cvr"
            roi_dir = ses_dir / "roi2bold"

            cvr_mag_path = cvr_dir / "CVR_mag.nii"
            cvr_delay_path = cvr_dir / "CVR_delay.nii"

            if not cvr_mag_path.exists() or not cvr_delay_path.exists():
                n_skipped_missing += 1
                if args.verbose:
                    print(f"Skipping {sub} {ses} (missing CVR files)")
                continue

            any_roi_found = False
            for roi_name, roi_file in ROI_FILES.items():
                roi_path = roi_dir / roi_file
                if not roi_path.exists():
                    if args.verbose:
                        print(f"Skipping {sub} {ses} {roi_name} (missing ROI file)")
                    continue

                try:
                    voxels_cvr_mag, voxels_cvr_delay = collect_session_voxels(
                        cvr_mag_path, cvr_delay_path, roi_path, mask_thr=args.mask_thr
                    )

                except Exception as e:
                    print(f"Error processing {sub} {ses} {roi_name}: {e}")
                    continue

                if voxels_cvr_mag.size == 0:
                    if args.verbose:
                        print(f"No valid voxels for {sub} {ses} {roi_name}")
                        continue
                
                pooled[roi_name]["cvr_mag"].extend(voxels_cvr_mag.tolist())
                pooled[roi_name]["cvr_delay"].extend(voxels_cvr_delay.tolist())
                any_roi_found = True

                if args.verbose:
                    print(f"Processed {sub} {ses} {roi_name}: {len(voxels_cvr_mag)} voxels")
            if any_roi_found:
                n_used += 1
            else:
                n_skipped_missing += 1
    
    for roi_key in ROI_FILES.keys():
        cvr_mag_list = pooled[roi_key]["cvr_mag"]
        cvr_delay_list = pooled[roi_key]["cvr_delay"]  

        if len(cvr_mag_list) == 0:
            print(f"Warning: No voxels collected for ROI {roi_key}")
            continue

        cvr_mags = np.array(cvr_mag_list, dtype=np.float32).reshape(-1, 1)
        cvr_delays = np.array(cvr_delay_list, dtype=np.float32).reshape(-1, 1)

        out_path = out_dir / f"HV_{roi_key}_cvr_dist.mat"
        savemat(out_path.as_posix(), {"cvr_mag": cvr_mags, "cvr_delay": cvr_delays})

        print(f"Saved {len(cvr_mags)} voxels for ROI {roi_key} to {out_path}")
    print(f"Total sessions used: {n_used}")
    print(f"Total sessions skipped (excluded): {n_skipped_excl}")
    print(f"Total sessions skipped (missing files): {n_skipped_missing}")

if __name__ == "__main__":
    main()
