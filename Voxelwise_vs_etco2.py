import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


bold_path = "/bids_dir/sub-01/func/boldmcf.nii"
etco2_path = "/Users/mac/PycharmProjects/pythonMPhysproject/EtCO2_results.txt"

TR = 1.55

bold_img = nib.load(bold_path)
bold = bold_img.get_fdata()
nx, ny, nz, nt = bold.shape
t_bold = np.arange(nt) * TR


etdf = pd.read_csv(etco2_path, sep=None, engine="python")
time_et = etdf["sec"].values

co2_cols = [c for c in etdf.columns if "etco2" in c.lower()]
co2_col = co2_cols[0]
et_raw = etdf[co2_col].values

pct_to_mmhg = 760.0 / 100.0
if np.median(et_raw) < 30:
    et_mmhg = et_raw * pct_to_mmhg
else:
    et_mmhg = et_raw.copy()


et_resampled = np.interp(t_bold, time_et, et_mmhg)


def zscore(x):
    s = np.std(x)
    if s < 1e-6:
        return np.zeros_like(x)
    return (x - np.mean(x)) / s


x0, y0, z0 = 47, 47, 25
voxel_ts = bold[x0, y0, z0, :]


plt.ion()
fig, ax = plt.subplots(figsize=(12, 5))
plt.subplots_adjust(bottom=0.25)

initial_shift = 0

et_shifted = np.interp(t_bold - initial_shift, time_et, et_mmhg)

[line_bold] = ax.plot(t_bold, zscore(voxel_ts), label="BOLD voxel", linewidth=2)

[line_co2] = ax.plot(t_bold, zscore(et_shifted),
                     label=f"EtCO₂ (shift = {initial_shift:.1f}s)", linewidth=2, alpha=0.8)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Normalized amplitude")
ax.set_title(f"BOLD vs EtCO₂ — Voxel ({x0},{y0},{z0})")
ax.grid(True)
ax.legend()


axcolor = 'lightgray'
ax_shift = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor=axcolor)

slider_shift = Slider(
    ax=ax_shift,
    label="Time shift (s)",
    valmin=-700,
    valmax=100,
    valinit=0,
    valstep=0.5
)

def update(val):
    shift = slider_shift.val

    et_shifted = np.interp(t_bold - shift, time_et, et_mmhg)

    line_co2.set_ydata(zscore(et_shifted))
    line_co2.set_label(f"EtCO₂ (shift = {shift:.1f}s)")

    ax.legend()
    fig.canvas.draw_idle()


slider_shift.on_changed(update)

plt.show(block=True)
