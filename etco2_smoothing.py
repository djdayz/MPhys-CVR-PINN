import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

etco2_path = "sub-15/ses-02/pre/EtCO2_mmHg.txt"

TR = 1.55
global_shift = 537.0
smooth_sec = 8.0
grad_ramp_thr = 0.3
normo_range = (34.5, 42)
hyper_range = (46.5, 55)

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


df = pd.read_csv(etco2_path, sep=None, engine="python")

time_et = df["sec"].values
co2_col = [c for c in df.columns if "etco2" in c.lower()][0]
et = df[co2_col].values

time_et_shifted = time_et - global_shift

t_bold = np.arange(
    int((time_et_shifted.max() - time_et_shifted.min()) / TR)
) * TR

et_on_bold = np.interp(t_bold, time_et_shifted, et, left=np.nan, right=np.nan)

valid = np.isfinite(et_on_bold)
et_fill = et_on_bold.copy()
idx = np.where(valid)[0]
et_fill[~valid] = np.interp(np.where(~valid)[0], idx, et_on_bold[idx])


smooth_win = int(np.ceil(smooth_sec / TR))
et_smooth = moving_average(et_fill, smooth_win)
et_smooth[~valid] = np.nan

grad = np.gradient(et_smooth, TR)


stable = valid & (np.abs(grad) <= grad_ramp_thr)
stable_normo = stable & (et_smooth >= normo_range[0]) & (et_smooth <= normo_range[1])
stable_hyper = stable & (et_smooth >= hyper_range[0]) & (et_smooth <= hyper_range[1])

fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axs[0].plot(t_bold, et_on_bold, color="gray", alpha=0.5, label="Raw EtCO2")
axs[0].plot(t_bold, et_smooth, color="black", linewidth=2, label="Smoothed EtCO2")
axs[0].axhspan(*normo_range, color="blue", alpha=0.1, label="Normo range")
axs[0].axhspan(*hyper_range, color="red", alpha=0.1, label="Hyper range")
axs[0].set_ylabel("EtCO2 (mmHg)")
axs[0].legend(loc="upper right")

axs[1].plot(t_bold, np.abs(grad), color="purple")
axs[1].axhline(grad_ramp_thr, color="red", linestyle="--", label="Ramp threshold")
axs[1].set_ylabel("|dEtCO2/dt| (mmHg/s)")
axs[1].legend()

axs[2].plot(t_bold, et_smooth, color="black", linewidth=1)
axs[2].scatter(t_bold[stable_normo], et_smooth[stable_normo],
               color="blue", s=10, label="Stable normo")
axs[2].scatter(t_bold[stable_hyper], et_smooth[stable_hyper],
               color="red", s=10, label="Stable hyper")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("EtCO2 (mmHg)")
axs[2].legend()

plt.tight_layout()
plt.show()
