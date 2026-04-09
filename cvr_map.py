import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.linalg import pinv
from scipy.stats import t as t_dist
from scipy.stats import norm
import matplotlib.pyplot as plt

bold_path = "sub-15/ses-02/pre/boldmcf.nii"
mask_path = "sub-15/ses-02/pre/boldmask_ero.nii"
etco2_path = "sub-15/ses-02/pre/EtCO2_mmHg.txt"
outdir = "sub-15/ses-02/cvr"
os.makedirs(outdir, exist_ok=True)

TR = 1.55
baseline_vols = 30

lag_min = -31.0
lag_max = 93.0
lag_step = TR

global_shift = 537.0
delay_samplingline = 4.0

threshold = 47.5
transition_sec = 60.0
transition_vols = int(np.ceil(transition_sec / TR))


bold_img = nib.load(bold_path)
bold = bold_img.get_fdata()
nx, ny, nz, nt = bold.shape
V = nx * ny * nz
affine = bold_img.affine

print("BOLD shape:", bold.shape)

mask_img = nib.load(mask_path)
mask_data = mask_img.get_fdata()

print("Mask raw shape:", mask_data.shape)

if mask_data.ndim == 4:
    mask3d = mask_data[..., 0]
else:
    mask3d = mask_data

mask = (mask3d > 0).astype(bool)


if mask.shape != bold[...,0].shape:
    raise ValueError(f"Mask shape {mask.shape} does not match BOLD spatial dimension {bold[...,0].shape}")

mask_flat = mask.reshape(-1)
Nmask = np.sum(mask_flat)

print(f"Brain mask voxels: {Nmask} / {V}")

bold_flat = bold.reshape((V, nt))
bold_flat_masked = bold_flat[mask_flat]

bold_baseline_map = bold[..., :baseline_vols].mean(axis=3)
global_baseline = bold_baseline_map[mask].mean()

print("Global BOLD baseline =", global_baseline)

Y = bold_flat_masked.T

Y_mean = Y.mean(axis=0)
sst = np.sum((Y - Y_mean)**2, axis=0)

t = np.arange(nt) * TR

df = pd.read_csv(etco2_path, sep=None, engine="python")
time_et = df["sec"].values
co2_col = [c for c in df.columns if "etco2_interp" in c.lower()][0]
et = df[co2_col].values
print("Using EtCO2 column:", co2_col)

time_et_shifted = time_et - global_shift

baseline_end = baseline_vols * TR
mask_et_baseline = (time_et_shifted >= 0) & (time_et_shifted <= baseline_end)

if np.sum(mask_et_baseline) == 0:
    et_baseline = et[:20].mean()
else:
    et_baseline = et[mask_et_baseline].mean()

print("EtCO2 baseline =", et_baseline)


lag_grid = np.arange(lag_min, lag_max + lag_step/2, lag_step)

best_ssr = np.full(Nmask, np.inf)
best_beta = np.full(Nmask, np.nan)
best_lag = np.full(Nmask, np.nan)
best_dof = np.zeros(Nmask)
best_sigma2 = np.full(Nmask, np.nan)
best_var_beta_et = np.full(Nmask, np.nan)

print("Running voxelwise GLM")

drift = np.arange(nt)

def moving_average(x, win):

    win = int(win)
    if win < 1:
        return x.copy()
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win) / win
    return np.convolve(xpad, kernel, mode="valid")

def stable_windows_from_etco2_gradient(t_bold, time_et_shifted, et, TR, smooth_sec=8.0, grad_ramp_thr=0.3, normo_range=(34.5, 42), hyper_range=(46.5, 55)):

    mean_bold = np.mean(bold_flat_masked, axis=0)

    et_on_bold = np.interp(t_bold, time_et_shifted, et, left=np.nan, right=np.nan)
    valid = np.isfinite(et_on_bold)

    corr = np.corrcoef(mean_bold[valid], et_on_bold[valid])[0, 1]
    print(f"corr(mean BOLD, EtCO2) = {corr}")

    plt.figure(figsize=(12, 4))
    plt.plot(t, (mean_bold - np.mean(mean_bold)) / np.std(mean_bold), label="Mean BOLD (z)")
    plt.plot(t, (et_on_bold - np.nanmean(et_on_bold)) / np.nanstd(et_on_bold), label="EtCO2 (z)")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.title("Alignment check: mean BOLD vs EtCO2")
    plt.show()

    if not np.any(valid):
        raise ValueError("No valid EtCO2 samples after interpolation onto BOLD grid.")

    et_fill = et_on_bold.copy()
    idx = np.where(valid)[0]
    et_fill[~valid] = np.interp(np.where(~valid)[0], idx, et_on_bold[idx])

    smooth_win = int(np.ceil(smooth_sec / TR))
    et_smooth = moving_average(et_fill, smooth_win)
    et_smooth[~valid] = np.nan

    grad = np.gradient(et_smooth, TR)

    stable = valid & np.isfinite(grad) & (np.abs(grad) <= grad_ramp_thr)

    normo_lo, normo_hi = normo_range
    hyper_lo, hyper_hi = hyper_range

    stable_normo = stable & (et_smooth >= normo_lo) & (et_smooth <= normo_hi)
    stable_hyper = stable & (et_smooth >= hyper_lo) & (et_smooth <= hyper_hi)

    info = {
        "et_on_bold": et_on_bold,
        "et_smooth": et_smooth,
        "grad": grad,
        "stable": stable,
        "stable_normo_n": int(stable_normo.sum()),
        "stable_hyper_n": int(stable_hyper.sum()),
        "normo_mean": float(np.nanmean(et_smooth[stable_normo])) if np.any(stable_normo) else np.nan,
        "hyper_mean": float(np.nanmean(et_smooth[stable_hyper])) if np.any(stable_hyper) else np.nan,
        "delta_mean": (float(np.nanmean(et_smooth[stable_hyper])) - float(np.nanmean(et_smooth[stable_normo])))
                      if (np.any(stable_normo) and np.any(stable_hyper)) else np.nan,
        "smooth_win_TRs": smooth_win,
    }
    return stable_normo, stable_hyper, info


def compute_tcnr_voxelwise(bold_flat_masked, stable_normo, stable_hyper, eps=1e-8):

    if stable_normo.sum() < 5 or stable_hyper.sum() < 5:
        raise ValueError(f"Not enough stable samples: normo={stable_normo.sum()}, hyper={stable_hyper.sum()}")

    baseline_voxel = np.mean(bold_flat_masked[:, :baseline_vols], axis=1)
    B_psc = 100.0 * (bold_flat_masked / (baseline_voxel[:, None] + 1e-8) - 1.0)

    mu_hyper = np.mean(B_psc[:, stable_hyper], axis=1)
    mu_normo = np.mean(B_psc[:, stable_normo], axis=1)
    sigma_base = np.std(B_psc[:, stable_normo], axis=1)
    good = sigma_base > 1e-3

    tcnr_masked = np.full(Nmask, np.nan)
    tcnr_masked[good] = (mu_hyper[good] - mu_normo[good]) / (sigma_base[good] + 1e-8)

    print("sigma_base min/median/max:",
          np.nanmin(sigma_base), np.nanmedian(sigma_base), np.nanmax(sigma_base))

    print("fraction sigma_base < 1e-6:", np.mean(sigma_base < 1e-6))
    print("fraction sigma_base == 0:", np.mean(sigma_base == 0))

    return tcnr_masked

stable_normo, stable_hyper, info = stable_windows_from_etco2_gradient(t_bold=t, time_et_shifted=time_et_shifted, et=et,
                                                                      TR=TR, smooth_sec=8.0, grad_ramp_thr=0.3,
                                                                      normo_range=(34.5, 42),
                                                                      hyper_range=(46.5, 55))

print("Stable normo vols:", info["stable_normo_n"], "mean EtCO2:", info["normo_mean"])
print("Stable hyper vols:", info["stable_hyper_n"], "mean EtCO2:", info["hyper_mean"])
print("delta mean EtCO2:", info["delta_mean"])
print("Smoothing window (TRs):", info["smooth_win_TRs"])
print("stable_normo count:", stable_normo.sum())
print("stable_hyper count:", stable_hyper.sum())
print("overlap count:", np.sum(stable_normo & stable_hyper))
print("valid count:", np.sum(np.isfinite(np.interp(t, time_et_shifted, et, left=np.nan, right=np.nan))))


tcnr_masked = compute_tcnr_voxelwise(bold_flat_masked, stable_normo, stable_hyper)
tcnr_pos = np.maximum(tcnr_masked, 0)

vals = tcnr_masked[np.isfinite(tcnr_masked)]
print("tCNR min/median/max:", np.min(vals), np.median(vals), np.max(vals))
print("tCNR percentiles 50/90/95/98/99:", np.percentile(vals, [50, 90, 95, 98, 99]))
print("fraction tCNR < 0:", np.mean(vals < 0))


for lag in lag_grid:

    et_shift = np.interp(t - lag, time_et_shifted, et, left=np.nan, right=np.nan)
    dEt = et_shift - et_baseline

    valid_t = ~np.isnan(dEt)
    if np.sum(valid_t) < nt/2:
        print(f"lag {lag:+.1f}: skipping (insufficient ETCO2 samples)")
        continue

    X = np.column_stack([np.ones(np.sum(valid_t)), dEt[valid_t], drift[valid_t]])
    pinvX = pinv(X)
    XtX_inv = pinv(X.T @ X)
    var_beta_et = XtX_inv[1, 1]

    Y_use = Y[valid_t, :]
    betas = pinvX @ Y_use

    Yhat = X @ betas
    resid = Y_use - Yhat
    ssr = np.sum(resid**2, axis=0)

    improved = ssr < best_ssr
    best_ssr[improved] = ssr[improved]
    best_beta[improved] = betas[1, improved]
    best_lag[improved] = lag
    best_var_beta_et[improved] = var_beta_et

    print(f"lag={lag:+.1f}s updated={np.sum(improved)} voxels")

    dof = np.sum(valid_t) - X.shape[1]
    sigma2 = ssr / dof

    best_sigma2[improved] = sigma2[improved]
    best_dof[improved] = dof

tiny = 1e-12

safe_sigma2 = np.maximum(best_sigma2, tiny)
safe_varb = np.maximum(best_var_beta_et, tiny)

SE_beta = np.sqrt(safe_sigma2 * safe_varb)

SE_beta = np.maximum(SE_beta, tiny)

t_masked = best_beta / SE_beta

df_safe = np.maximum(best_dof, 1)
p_vals = 2.0 * (1.0 - t_dist.cdf(np.abs(t_masked), df=df_safe))

p_min = np.finfo(float).tiny
p_vals = np.clip(p_vals, p_min, 1.0)
z_masked = norm.isf(p_vals / 2.0) * np.sign(t_masked)

z_full = np.zeros(V)
z_full[mask_flat] = z_masked
z_vol = z_full.reshape(nx, ny, nz)

print("z min/max:", np.nanmin(z_masked), np.nanmax(z_masked))
print("fraction z > 3:", np.mean(z_masked > 3))

R2_masked = 1.0 - (best_ssr / sst)
R2_masked = np.clip(R2_masked, 0.0, 1.0)

R2_full = np.zeros(V)
R2_full[mask_flat] = R2_masked
R2_vol = R2_full.reshape(nx, ny, nz)

CVR_masked = (best_beta / global_baseline) * 100.0
delay_masked = best_lag + delay_samplingline
SE_cvr_masked = (SE_beta / global_baseline) * 100.0


cvr_vals = CVR_masked[np.isfinite(CVR_masked)]

cvr_mean = np.mean(cvr_vals)
cvr_std = np.std(cvr_vals)

print("CVR mean:", cvr_mean)
print("CVR std:", cvr_std)

z_cvr_dist_masked = (CVR_masked - cvr_mean) / (cvr_std)

CVR_clipped_masked = np.clip(CVR_masked, 0.0, 0.5)
delay_clipped_masked = np.clip(delay_masked, 0.0, 80.0)

CVR_full = np.zeros(V)
CVR_clipped_full = np.zeros(V)
delay_full = np.zeros(V)
delay_clipped_full = np.zeros(V)
tcnr_full = np.zeros(V)
SE_cvr_full = np.zeros(V)
tcnr_pos_full = np.zeros(V)

CVR_full[mask_flat] = CVR_masked
CVR_clipped_full[mask_flat] = CVR_clipped_masked
delay_full[mask_flat] = delay_masked
delay_clipped_full[mask_flat] = delay_clipped_masked
tcnr_full[mask_flat] = tcnr_masked
SE_cvr_full[mask_flat] = SE_cvr_masked
tcnr_pos_full[mask_flat] = tcnr_pos

CVR_vol = CVR_full.reshape(nx, ny, nz)
CVR_clipped_vol = CVR_clipped_full.reshape(nx, ny, nz)
delay_vol = delay_full.reshape(nx, ny, nz)
delay_clipped_vol = delay_clipped_full.reshape(nx, ny, nz)
tcnr_vol = tcnr_full.reshape(nx, ny, nz)
SE_cvr_vol = SE_cvr_full.reshape(nx, ny, nz)
tcnr_pos_vol = tcnr_pos_full.reshape(nx, ny, nz)

z_cvr_dist_full = np.zeros(V)
z_cvr_dist_full[mask_flat] = z_cvr_dist_masked
z_cvr_dist_vol = z_cvr_dist_full.reshape(nx, ny, nz)

def save_map(vol, name):
    nib.save(nib.Nifti1Image(vol.astype(np.float32), affine),
             os.path.join(outdir, name))
    print("Saved:", name)

save_map(CVR_vol, "CVR_mag.nii")
save_map(delay_vol, "CVR_delay.nii")
save_map(CVR_clipped_vol, "CVR_mag_clipped.nii")
save_map(delay_clipped_vol, "CVR_delay_clipped.nii")
save_map(tcnr_vol, "tCNR_masked.nii")
save_map(R2_vol, "CVR_R2_masked.nii")
save_map(z_vol, "CVR_zstat_masked.nii")
save_map(z_cvr_dist_vol, "CVR_zscore_distribution.nii")
save_map(SE_cvr_vol, "CVR_SE.nii")
save_map(tcnr_pos_vol, "tCNR_pos.nii")


print("\nDONE; masked CVR pipeline complete.")
