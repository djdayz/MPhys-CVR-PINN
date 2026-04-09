
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


X_CVR = np.arange(-0.5, 2.0001, 0.01, dtype=np.float32)
X_DEL = np.arange(0, 201, 1, dtype=np.int32)

ROIS = ["cgm", "sgm", "wm", "vcsf", "vessel"]


def _load_pdf(pdf_path):
    d = loadmat(pdf_path.as_posix())
    if "w_cvr" not in d or "w_delays" not in d:
        raise KeyError(f"{pdf_path} must contain variables 'w_cvr' and 'w_delays'")
    w_cvr = np.asarray(d["w_cvr"]).squeeze().astype(np.float64)
    w_del = np.asarray(d["w_delays"]).squeeze().astype(np.float64)

    if w_cvr.size != X_CVR.size:
        raise ValueError(f"{pdf_path}: w_cvr length {w_cvr.size} != {X_CVR.size} (grid mismatch)")
    if w_del.size != X_DEL.size:
        raise ValueError(f"{pdf_path}: w_delays length {w_del.size} != {X_DEL.size} (grid mismatch)")

    w_cvr = w_cvr / (w_cvr.sum() + 1e-12)
    w_del = w_del / (w_del.sum() + 1e-12)

    return w_cvr, w_del


def _sample_from_weights(x, w, n, rng):
    idx = rng.choice(len(x), size=int(n), replace=True, p=w)
    return x[idx]


def parse_nvox_pairs(pairs):

    out: Dict[str, int] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Bad --nvox pair '{p}', expected key=value")
        k, v = p.split("=", 1)
        k = k.strip().lower()
        v = int(v.strip())
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Create CVR_distribution_{ROI}_{Nvox}voxels.mat by sampling from Probability density PDFs."
    )
    ap.add_argument("--pdf_dir", type=Path, default=Path("hv_dist/pde"), help="Directory containing pdf_{roi}.mat")
    ap.add_argument("--out_dir", type=Path, default=Path("hv_dist/cvr_dist"), help="Output directory")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    ap.add_argument("--delay_max", type=int, default=85, help="Optional hard cap on sampled delays (<=200). Default 200.")
    ap.add_argument("--nvox_json", type=Path, default=Path("nvox.json"), help="JSON mapping ROI->Nvox, e.g. {'cgm':123,...}")
    ap.add_argument("--nvox", nargs="*", default=[], help="Override/define Nvox as pairs, e.g. cgm=1000 wm=2000")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    nvox: Dict[str, int] = {}

    if args.nvox_json is not None:
        nvox.update(json.loads(args.nvox_json.read_text()))

    if args.nvox:
        nvox.update(parse_nvox_pairs(args.nvox))

    missing = [r for r in ROIS if r not in nvox]
    if missing:
        raise SystemExit(
            "Missing Nvox for: " + ", ".join(missing) +
            "\nProvide --nvox_json or --nvox pairs for all rois: " + ", ".join(ROIS)
        )

    for roi in ROIS:
        pdf_path = args.pdf_dir / f"pdf_{roi}.mat"
        if not pdf_path.exists():
            raise SystemExit(f"Missing {pdf_path}")

        w_cvr, w_del = _load_pdf(pdf_path)
        N = int(nvox[roi])

        cvr_dist = _sample_from_weights(X_CVR, w_cvr, N, rng).astype(np.float32)
        del_dist = _sample_from_weights(X_DEL.astype(np.float32), w_del, N, rng)
        del_dist = np.rint(del_dist).astype(np.int32)

        if args.delay_max < 200:
            del_dist = np.clip(del_dist, 0, int(args.delay_max))

        out = {
            "CVR_magnitude_distribution": cvr_dist.reshape(-1, 1),
            "CVR_delay_distribution": del_dist.reshape(-1, 1),
            "CVR_magnitude_mean": float(np.mean(cvr_dist)),
            "CVR_magnitude_std": float(np.std(cvr_dist, ddof=0)),
            "CVR_delay_mean": float(np.mean(del_dist)),
            "CVR_delay_std": float(np.std(del_dist, ddof=0)),
        }

        out_path = args.out_dir / f"CVR_dist_{roi}_{N}voxels.mat"
        savemat(out_path.as_posix(), out)
        print(f"Saved {out_path}")

    (args.out_dir / "nvox_used.json").write_text(json.dumps(nvox, indent=2))
    print(f"Saved {args.out_dir / 'nvox_used.json'}")


if __name__ == "__main__":
    main()