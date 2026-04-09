
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.neighbors import KernelDensity


ROIS = ["cgm", "sgm", "wm", "vcsf", "vessel"]

BW = {
    "cgm": {"cvr": 0.013, "delay": 1.04},
    "sgm": {"cvr": 0.015, "delay": 1.58},
    "wm": {"cvr": 0.006, "delay": 1.97},
    "vcsf": {"cvr": 0.035, "delay": 4.36},
    "vessel": {"cvr": 0.087, "delay": 0.69}
}


def kde_curve(samples, grid, bw):
    kde = KernelDensity(kernel="gaussian", bandwidth=bw)
    kde.fit(samples.reshape(-1, 1))
    logp = kde.score_samples(grid.reshape(-1, 1))
    return np.exp(logp)


def plot_roi(roi, hv_dir, out_dir):
    mat_path = hv_dir / f"HV_{roi}_cvr_dist.mat"
    d = loadmat(mat_path.as_posix())
    cvr = np.asarray(d["cvr_mag"]).squeeze()
    delays = np.asarray(d["cvr_delay"]).squeeze()

    delays = delays[(delays > 0) & (delays <= 85)]

    MAX = 50000
    if cvr.size > MAX:
        cvr = np.random.choice(cvr, MAX, replace=False)
    if delays.size > MAX:
        delays = np.random.choice(delays, MAX, replace=False)

    grid_cvr = np.linspace(cvr.min(), cvr.max(), 500)
    pdf_cvr = kde_curve(cvr, grid_cvr, BW[roi]["cvr"])

    plt.figure(figsize=(6, 4))
    plt.hist(cvr, bins=100, density=True, alpha=0.4)
    plt.plot(grid_cvr, pdf_cvr, linewidth=2)
    plt.title(f"{roi} CVR KDE")
    plt.xlabel("CVR magnitude (%/mmHg)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_dir / f"{roi}_cvr_kde.png", dpi=300)
    plt.close()

    grid_delay = np.linspace(0, 90, 500)
    pdf_delay = kde_curve(delays, grid_delay, BW[roi]["delay"])

    plt.figure(figsize=(6, 4))
    plt.hist(delays, bins=100, density=True, alpha=0.4)
    plt.plot(grid_delay, pdf_delay, linewidth=2)
    plt.title(f"{roi} Delay KDE")
    plt.xlabel("Delay (s)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_dir / f"{roi}_delay_kde.png", dpi=300)
    plt.close()

    print(f"Plotted {roi}")


def main():
    hv_dir = Path("hv_dist")
    out_dir = Path("hv_dist/kde_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    for roi in ROIS:
        plot_roi(roi, hv_dir, out_dir)


if __name__ == "__main__":
    main()