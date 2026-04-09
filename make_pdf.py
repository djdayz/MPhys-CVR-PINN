
import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from sklearn.neighbors import KernelDensity


ROIS = ["cgm", "sgm", "wm", "vcsf", "vessel"]

X_DELAYS = np.arange(0, 201, 1, dtype=np.float64)
X_CVR = np.arange(-0.5, 2.0001, 0.01, dtype=np.float64)


def kde_pdf(samples, x_grid, bandwidth):

    samples = np.asarray(samples, dtype=np.float64).reshape(-1, 1)
    xg = np.asarray(x_grid, dtype=np.float64).reshape(-1, 1)

    kde = KernelDensity(kernel="gaussian", bandwidth=float(bandwidth))
    kde.fit(samples)

    logp = kde.score_samples(xg)
    p = np.exp(logp)

    p = p / (p.sum() + 1e-12)
    return p.astype(np.float32)


def load_hv_dist(mat_path):

    d = loadmat(mat_path.as_posix())
    if "cvr_mag" not in d or "cvr_delay" not in d:
        raise KeyError(f"{mat_path} must contain variables 'CVR' and 'delays'")

    cvr = np.asarray(d["cvr_mag"]).squeeze().astype(np.float64)
    delays = np.asarray(d["cvr_delay"]).squeeze().astype(np.float64)
    return cvr, delays


def main():
    ap = argparse.ArgumentParser(
        description="Build Probability_density_function_{ROI}.mat from HV_{ROI}_CVR_distribution.mat (KDE for CVR and delays)."
    )
    ap.add_argument("--hv_dir", type=Path, default=Path("hv_dist"), help="Directory containing HV_{ROI}_CVR_distribution.mat")
    ap.add_argument("--out_dir", type=Path, default=Path("hv_dist/pde"), help="Output directory for Probability_density_function_{ROI}.mat")
    ap.add_argument("--delay_bw", type=float, default=2.0, help="KDE bandwidth for delays (seconds)")
    ap.add_argument("--cvr_bw", type=float, default=0.03, help="KDE bandwidth for CVR magnitude (%/mmHg)")
    ap.add_argument(
        "--roi_bandwidths",
        action="store_true",
        help="Use different default bandwidths per ROI (recommended). Overrides --delay_bw/--cvr_bw defaults.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    roi_bw: Dict[str, Tuple[float, float]] = {
        "cgm": (1.04, 0.013),
        "sgm": (1.58, 0.015),
        "wm": (1.97, 0.006),
        "vcsf": (4.36, 0.035),
        "vessel": (0.69, 0.087),
    }

    for roi in ROIS:
        in_path = args.hv_dir / f"HV_{roi}_cvr_dist.mat"
        if not in_path.exists():
            print(f"[SKIP] Missing {in_path}")
            continue

        cvr, delays = load_hv_dist(in_path)

        delays_pos = delays[delays > 0]
        if delays_pos.size < 10 or cvr.size < 10:
            print(f"[SKIP] Too few samples in {in_path} (CVR n={cvr.size}, delays_pos n={delays_pos.size})")
            continue

        if args.roi_bandwidths:
            delay_bw, cvr_bw = roi_bw.get(roi, (args.delay_bw, args.cvr_bw))
        else:
            delay_bw, cvr_bw = args.delay_bw, args.cvr_bw

        w_delays = kde_pdf(delays_pos, X_DELAYS, bandwidth=float(delay_bw))
        w_cvr = kde_pdf(cvr, X_CVR, bandwidth=float(cvr_bw))

        out_path = args.out_dir / f"pdf_{roi}.mat"
        savemat(out_path.as_posix(), {"w_cvr": w_cvr, "w_delays": w_delays})

        print(
            f"Saved {out_path} | "
            f"delay_bw={delay_bw}, cvr_bw={cvr_bw} | "
            f"CVR n={cvr.size}, delays_pos n={delays_pos.size}"
        )


if __name__ == "__main__":
    main()