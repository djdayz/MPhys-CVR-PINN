import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt


EXCLUDE = {
    ("sub-04", "ses-01"),
    ("sub-04", "ses-02"),
    ("sub-11", "ses-01"),
    ("sub-12", "ses-01"),
    ("sub-12", "ses-02"),
    ("sub-13", "ses-01"),
    ("sub-13", "ses-02"),
}

ROI_MASKS = {
    "cgm": "cortical_gm_mask_in_BOLD_bin.nii",
    "sgm": "subcort_gm_mask_in_BOLD_bin_ero1.nii",
    "wm":  "wm_mask_in_BOLD_bin_ero1.nii",
    "vcsf": "vcsf_mask_in_BOLD_bin.nii",
    "vessel": "vessel_mask_in_BOLD_bin.nii",
}


def load_nii(path):
    return nib.load(str(path)).get_fdata()


def extract_vals_in_mask(map_path, mask_path):
    data = load_nii(map_path)
    mask = load_nii(mask_path) > 0.5
    vals = data[mask]
    vals = vals[np.isfinite(vals)]
    return vals


class EmpiricalSampler:

    def __init__(self, vals, eps=1e-6):
        vals = np.asarray(vals)
        vals = vals[np.isfinite(vals)]
        vals = vals[np.abs(vals) > eps]
        if vals.size == 0:
            raise ValueError("No valid values to sample from (after cleaning).")
        self.vals_sorted = np.sort(vals)
        self.n = self.vals_sorted.size

    def sample(self, size, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        idx = rng.integers(0, self.n, size=size)
        return self.vals_sorted[idx]


def parse_bins(bins_arg):

    if isinstance(bins_arg, int):
        return bins_arg
    if isinstance(bins_arg, str):
        s = bins_arg.strip().lower()
        if s.isdigit():
            return int(s)
        return s
    return bins_arg


def resolve_bins_for_data(x, bins_spec, lo, hi):

    if isinstance(bins_spec, int):
        return int(bins_spec)
    if isinstance(bins_spec, str):
        edges = np.histogram_bin_edges(x, bins=bins_spec, range=(lo, hi))
        return max(5, int(len(edges) - 1))
    return int(bins_spec)


def smooth_hist_density_line(x, bins, lo, hi, sigma_bins=2.5):

    hist, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    sigma_bins = float(sigma_bins)
    if sigma_bins <= 0:
        return centers, hist

    rad = int(np.ceil(4 * sigma_bins))
    kx = np.arange(-rad, rad + 1, dtype=float)
    kern = np.exp(-0.5 * (kx / sigma_bins) ** 2)
    kern /= kern.sum()

    hist_s = np.convolve(hist, kern, mode="same")
    return centers, hist_s

def compute_bins(x, bins, lo, hi, delay_binwidth=None):

    if delay_binwidth and delay_binwidth > 0:
        bw = float(delay_binwidth)
        start = np.floor(lo / bw) * bw
        stop  = np.ceil(hi / bw) * bw
        edges = np.arange(start, stop + 0.5 * bw, bw)
        if edges.size < 5:
            edges = np.linspace(lo, hi, 10)
        return edges

    if isinstance(bins, str):
        edges = np.histogram_bin_edges(x, bins=bins, range=(lo, hi))
        return edges
    else:
        return int(bins)


def kde_pdf(x, lo, hi, n_grid=1200, bw=None, max_points=20000, rng=None):

    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.array([]), np.array([])


    rng = np.random.default_rng() if rng is None else rng

    if x.size > max_points:
        x = rng.choice(x, size=max_points, replace=False)

    xgrid = np.linspace(lo, hi, n_grid)

    std = np.std(x, ddof=1) if x.size > 1 else 1.0
    n = x.size
    if bw is None:
        bw = 1.06 * std * (n ** (-1/5))
        if not np.isfinite(bw) or bw <= 0:
            bw = (hi - lo) / 50.0 if hi > lo else 1.0

    chunk = 4000
    pdf = np.zeros_like(xgrid)
    for i in range(0, x.size, chunk):
        xi = x[i:i+chunk]
        z = (xgrid[:, None] - x[None, :]) / bw
        pdf += np.mean(np.exp(-0.5 * z * z), axis=1) * (xi.size / x.size)

    pdf = pdf / (bw * np.sqrt(2 * np.pi))

    return xgrid, pdf

def hist_pdf_line(x, bins, lo, hi):

    hist, edges = np.histogram(x, bins=bins, range=None if isinstance(bins, np.ndarray) else (lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    return centers, hist

def plot_original_vs_sampled(vals, sampler, title, out_png, bins="fd", n_sample=200000, metric="mag", tr=1.55, delay_use_tr_bins=True, touch_bars=True, show_model_line=True, mag_kde_bw=None, delay_kde_bw=None):
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    vals = vals[np.abs(vals) > 1e-6]

    n_sample = min(n_sample, max(5000, vals.size))
    samp = sampler.sample(n_sample)

    lo = float(min(vals.min(), samp.min()))
    hi = float(max(vals.max(), samp.max()))

    delay_binwidth = tr if (metric == "delay" and delay_use_tr_bins) else None
    bins_used = compute_bins(vals, bins=bins, lo=lo, hi=hi, delay_binwidth=delay_binwidth)

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    hist_kw = dict(
        bins=bins_used,
        range=None if isinstance(bins_used, np.ndarray) else (lo, hi),
        density=True,
        alpha=0.6
    )
    if touch_bars:
        hist_kw.update(dict(histtype="stepfilled", edgecolor="none", linewidth=0.0))
    else:
        hist_kw.update(dict(edgecolor="black", linewidth=0.3))

    ax1.hist(vals, **hist_kw)
    if show_model_line:
        if metric == "mag":
            xc, yc = hist_pdf_line(vals, bins=bins_used, lo=lo, hi=hi)
            ax1.plot(xc, yc, linewidth=2, label="Empricial PDF histogram line")
            ax1.legend()
        else:
            xg, pdf = kde_pdf(vals, lo, hi, bw=delay_kde_bw)
            if xg.size:
                ax1.plot(xg, pdf, linewidth=2, label="KDE model line")
                ax1.legend()
    ax1.axvline(0, linewidth=1)
    ax1.set_title("Original (pooled voxels)")
    ax1.set_xlabel("CVR value")
    ax1.set_ylabel("Density")

    ax2.hist(samp, **hist_kw)
    if show_model_line:
        if metric == "mag":
            xg, pdf = kde_pdf(samp, lo, hi, bw=mag_kde_bw)
            if xg.size:
                ax2.plot(xg, pdf, linewidth=2, label="KDE model line")
                ax2.legend()
        else:
            xg, pdf = kde_pdf(samp, lo, hi, bw=delay_kde_bw)
            if xg.size:
                ax2.plot(xg, pdf, linewidth=2, label="KDE model line")
                ax2.legend()
    ax2.axvline(0, linewidth=1)
    ax2.set_title(f"Sampled ({n_sample:,})")
    ax2.set_xlabel("CVR value")

    ax3.hist(vals, **{**hist_kw, "alpha": 0.35, "label": "Original"})
    ax3.hist(samp, **{**hist_kw, "alpha": 0.35, "label": "Sampled"})
    if show_model_line:
        xg1, pdf1 = kde_pdf(vals, lo, hi, bw=(mag_kde_bw if metric=="mag" else delay_kde_bw))
        xg2, pdf2 = kde_pdf(samp, lo, hi, bw=(mag_kde_bw if metric=="mag" else delay_kde_bw))
        if xg1.size:
            ax3.plot(xg1, pdf1, linewidth=2, label="Orig KDE")
        if xg2.size:
            ax3.plot(xg2, pdf2, linewidth=2, label="Samp KDE")
    ax3.axvline(0, linewidth=1)
    ax3.set_title("Overlay check")
    ax3.set_xlabel("CVR value")
    ax3.legend()

    if metric == "mag":
        p_lo, p_hi = np.percentile(vals, [0.5, 99.5])
        ax1.set_xlim(p_lo, p_hi)
        ax2.set_xlim(p_lo, p_hi)
        ax3.set_xlim(p_lo, p_hi)

    fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_empirical_distributions(bids_dir, mag_eps=1e-6, keep_zero_delay=True):
    dist_vals = {(roi, "mag"): [] for roi in ROI_MASKS}
    dist_vals.update({(roi, "delay"): [] for roi in ROI_MASKS})
    qc_rows = []

    for sub_dir in sorted(bids_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        sub = sub_dir.name

        for ses_dir in sorted(sub_dir.glob("ses-*")):
            if not ses_dir.is_dir():
                continue
            ses = ses_dir.name

            if (sub, ses) in EXCLUDE:
                continue

            cvr_dir = ses_dir / "cvr"
            roi2bold_dir = ses_dir / "roi2bold"

            mag_path = cvr_dir / "CVR_mag.nii"
            delay_path = cvr_dir / "CVR_delay.nii"

            if not mag_path.exists() or not delay_path.exists():
                continue

            for roi, mask_name in ROI_MASKS.items():
                mask_path = roi2bold_dir / mask_name
                if not mask_path.exists():
                    continue

                mag_vals = extract_vals_in_mask(mag_path, mask_path)
                mag_vals = mag_vals[np.isfinite(mag_vals)]
                mag_vals = mag_vals[np.abs(mag_vals) > mag_eps]

                delay_vals = extract_vals_in_mask(delay_path, mask_path)
                delay_vals = delay_vals[np.isfinite(delay_vals)]
                if not keep_zero_delay:
                    delay_vals = delay_vals[delay_vals != 0]

                dist_vals[(roi, "mag")].append(mag_vals)
                dist_vals[(roi, "delay")].append(delay_vals)

                qc_rows.append({
                    "sub": sub, "ses": ses, "roi": roi,
                    "n_mag": int(mag_vals.size),
                    "n_delay": int(delay_vals.size),
                })

    for key in list(dist_vals.keys()):
        chunks = dist_vals[key]
        dist_vals[key] = np.concatenate(chunks) if len(chunks) else np.array([], dtype=float)

    meta_df = pd.DataFrame(qc_rows)
    return dist_vals, meta_df


def add_jitter_and_clip(samples, rng, jitter_sd=0.0, clip_pct=0.0, ref_vals=None):
    s = np.asarray(samples, dtype=np.float32)

    if jitter_sd and float(jitter_sd) > 0:
        s = s + rng.normal(loc=0.0, scale=float(jitter_sd), size=s.shape).astype(np.float32)

    if clip_pct and float(clip_pct) > 0:
        clip_pct = float(clip_pct)
        lo_p = clip_pct
        hi_p = 100.0 - clip_pct

        ref = s if ref_vals is None else np.asarray(ref_vals, dtype=np.float32)
        ref = ref[np.isfinite(ref)]
        if ref.size > 10:
            lo, hi = np.percentile(ref, [lo_p, hi_p])
            s = np.clip(s, lo, hi).astype(np.float32)

    return s


def apply_sampler_to_mida_mask(mida_ref_nii, mida_roi_mask_nii, sampler, seed, fill_value=0.0, jitter_sd=0.0, clip_pct=0.0, ref_vals_for_clip=None):
    ref_img = nib.load(str(mida_ref_nii))
    ref_data = ref_img.get_fdata()

    mask = nib.load(str(mida_roi_mask_nii)).get_fdata() > 0.5
    out = np.full(ref_data.shape, fill_value, dtype=np.float32)

    idx = np.where(mask.ravel())[0]
    n = idx.size
    if n == 0:
        raise ValueError(f"MIDA ROI mask has 0 voxels: {mida_roi_mask_nii}")

    rng = np.random.default_rng(seed)
    samples = sampler.sample(n, rng=rng).astype(np.float32)
    samples = add_jitter_and_clip(samples, rng,
                                  jitter_sd=jitter_sd,
                                  clip_pct=clip_pct,
                                  ref_vals=ref_vals_for_clip)

    out_flat = out.ravel()
    out_flat[idx] = samples
    out = out_flat.reshape(ref_data.shape)

    return nib.Nifti1Image(out, affine=ref_img.affine, header=ref_img.header)

def fill_volume_from_roi(volume, mask, samples, overwrite=True):

    m = mask.astype(bool)
    if m.sum() == 0:
        return 0

    if overwrite:
        volume[m] = samples
        return int(m.sum())

    empty = m & (np.abs(volume) < 1e-8)
    n = int(empty.sum())
    if n > 0:
        volume[empty] = samples[:n]
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids_dir", required=True, type=str)
    ap.add_argument("--out_dir", default="empirical_dist_outputs", type=str)

    ap.add_argument("--bins", default="fd", type=str,
                    help="Histogram bins: int like '120' OR a rule like 'fd'/'auto'/'sturges'.")

    ap.add_argument("--mag_eps", default=1e-6, type=float)
    ap.add_argument("--drop_zero_delay", action="store_true",
                    help="Drop delay==0 if you believe 0 codes invalid estimates.")

    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing output files (plots + MIDA NIfTIs).")

    ap.add_argument("--mida_ref", type=str, default=None,
                    help="Reference MIDA-space NIfTI (defines shape/affine/header).")
    ap.add_argument("--mida_mask_dir", type=str, default=None,
                    help="Directory containing MIDA ROI masks (cgm/sgm/wm/vcsf/vessel).")
    ap.add_argument("--mida_mask_pattern", type=str, default="{roi}_mask.nii",
                    help="Pattern for MIDA ROI masks, e.g. '{roi}_mask.nii'.")

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--delay_jitter_sd", type=float, default=0.8,
                    help="Gaussian jitter SD (seconds) added AFTER resampling delay values. "
                         "TR=1.55s => try 0.4–0.6s. Default 0.5s.")
    ap.add_argument("--mag_jitter_sd", type=float, default=0.0,
                    help="Gaussian jitter SD added AFTER resampling magnitude values. Default 0.")
    ap.add_argument("--clip_samples_pct", type=float, default=0.0,
                    help="Optional: clip jittered samples to [p, 100-p] percentiles of ROI distribution. "
                         "Example 0.5 clips to [0.5, 99.5]. 0 disables clipping.")

    args = ap.parse_args()

    bids_dir = Path(args.bids_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bins_spec = parse_bins(args.bins)
    keep_zero_delay = not args.drop_zero_delay

    dist_vals, qc_df = build_empirical_distributions(
        bids_dir=bids_dir,
        mag_eps=args.mag_eps,
        keep_zero_delay=keep_zero_delay,
    )

    qc_csv = out_dir / "qc_voxel_counts_per_subses.csv"
    if qc_csv.exists() and not args.overwrite:
        pass
    else:
        qc_df.to_csv(qc_csv, index=False)

    pooled_csv = out_dir / "pooled_empirical_distributions.csv"
    if not pooled_csv.exists() or args.overwrite:
        rows = []
        for (roi, metric), vals in dist_vals.items():
            for v in vals:
                rows.append({"roi": roi, "metric": metric, "value": float(v)})
        pooled_df = pd.DataFrame(rows)
        pooled_df.to_csv(pooled_csv, index=False)

    print(f"Saved/checked QC: {qc_csv}")
    print(f"Saved/checked pooled distributions: {pooled_csv}")

    samplers = {}
    for roi in ROI_MASKS.keys():
        for metric in ["mag", "delay"]:
            vals = dist_vals[(roi, metric)]
            if vals.size == 0:
                print(f"WARNING: no data for {roi}/{metric}")
                continue

            sampler = EmpiricalSampler(vals, eps=args.mag_eps)
            samplers[(roi, metric)] = sampler

            out_png = out_dir / f"{roi}_{metric}_orig_vs_sampled.png"
            title = f"{roi.upper()} {metric.upper()} — empirical (orig vs sampled)"

            model_sd = args.delay_jitter_sd if metric == "delay" else args.mag_jitter_sd

            plot_original_vs_sampled(
                vals=vals,
                sampler=sampler,
                title=title,
                out_png=out_png,
                bins=args.bins,
                metric=metric,
                tr=1.55,
                delay_use_tr_bins=True,
                touch_bars=True,
                show_model_line=True,
                mag_kde_bw=None,
                delay_kde_bw=0.8,
            )

            if out_png.exists():
                print(f"Saved/checked plot: {out_png}")

    if args.mida_ref and args.mida_mask_dir:
        mida_ref = Path(args.mida_ref)
        mask_dir = Path(args.mida_mask_dir)

        ref_img = nib.load(str(mida_ref))
        shape = ref_img.shape
        mag_vol = np.full(shape, 0.0, dtype=np.float32)
        delay_vol = np.full(shape, 0.0, dtype=np.float32)

        roi_priority = ["wm", "cgm", "sgm", "vcsf", "vessel"]

        for roi in roi_priority:
            mask_path = mask_dir / args.mida_mask_pattern.format(roi=roi)
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing MIDA mask for {roi}: {mask_path}")

            mask = nib.load(str(mask_path)).get_fdata() > 0.5
            nvox = int(mask.sum())
            if nvox == 0:
                print(f"WARNING: {roi} mask has 0 voxels in MIDA, skipping.")
                continue

            mag_sampler = samplers.get((roi, "mag"))
            if mag_sampler is None:
                raise RuntimeError(f"No sampler for {roi}/mag (empty distribution?)")

            rng_mag = np.random.default_rng(args.seed + 100 + roi_priority.index(roi))
            mag_samples = mag_sampler.sample(nvox, rng=rng_mag).astype(np.float32)

            if args.mag_jitter_sd and args.mag_jitter_sd > 0:
                mag_samples = add_jitter_and_clip(
                    mag_samples, rng_mag,
                    jitter_sd=args.mag_jitter_sd,
                    clip_pct=args.clip_samples_pct,
                    ref_vals=dist_vals[(roi, "mag")]
                )

            delay_sampler = samplers.get((roi, "delay"))
            if delay_sampler is None:
                raise RuntimeError(f"No sampler for {roi}/delay (empty distribution?)")

            rng_delay = np.random.default_rng(args.seed + 200 + roi_priority.index(roi))
            delay_samples = delay_sampler.sample(nvox, rng=rng_delay).astype(np.float32)

            if args.delay_jitter_sd and args.delay_jitter_sd > 0:
                delay_samples = add_jitter_and_clip(
                    delay_samples, rng_delay,
                    jitter_sd=args.delay_jitter_sd,
                    clip_pct=args.clip_samples_pct,
                    ref_vals=dist_vals[(roi, "delay")]
                )

            wrote_mag = fill_volume_from_roi(mag_vol, mask, mag_samples, overwrite=True)
            wrote_delay = fill_volume_from_roi(delay_vol, mask, delay_samples, overwrite=True)

            print(f"Filled ROI {roi}: mag {wrote_mag} vox, delay {wrote_delay} vox")

        out_mag = out_dir / "MIDA_synthetic_CVR_mag_empirical.nii.gz"
        out_delay = out_dir / "MIDA_synthetic_CVR_delay_empirical.nii.gz"

        if not args.overwrite:
            if out_mag.exists():
                raise FileExistsError(f"{out_mag} exists. Use --overwrite to replace.")
            if out_delay.exists():
                raise FileExistsError(f"{out_delay} exists. Use --overwrite to replace.")

        nib.save(nib.Nifti1Image(mag_vol, ref_img.affine, ref_img.header), str(out_mag))
        nib.save(nib.Nifti1Image(delay_vol, ref_img.affine, ref_img.header), str(out_delay))

        print(f"Saved MIDA synthetic mag:   {out_mag}")
        print(f"Saved MIDA synthetic delay: {out_delay}")
        print(f"Delay jitter used: {args.delay_jitter_sd:.3g} s (TR=1.55s)")
    else:
        print("MIDA step skipped (provide --mida_ref and --mida_mask_dir).")


if __name__ == "__main__":
    main()
