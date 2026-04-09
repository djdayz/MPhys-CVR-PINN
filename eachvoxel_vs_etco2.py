import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

bold_path = "bids_dir/sub-01/ses-02/pre/boldmcf.nii"
etco2_path = "bids_dir/sub-01/ses-02/pre/EtCO2_mmHg.txt"

TR = 1.55

bold_img = nib.load(bold_path)
bold = bold_img.get_fdata()
nx, ny, nz, nt = bold.shape
V = nx * ny * nz
t_bold = np.arange(nt) * TR

bold_flat = bold.reshape(V, nt)

etdf = pd.read_csv(etco2_path, sep=None, engine="python")
time_et = etdf["sec"].values

co2_cols = [c for c in etdf.columns if "etco2_interp" in c.lower()]
co2_col = co2_cols[0]
et_raw = etdf[co2_col].values

et_resampled = np.interp(t_bold, time_et, et_raw)


def zscore(x):
    s = np.std(x)
    if s < 1e-6:
        return np.zeros_like(x)
    return (x - np.mean(x)) / s


initial_voxel = 0
voxel_ts = bold_flat[initial_voxel, :]


plt.ion()
fig, ax = plt.subplots(figsize=(12, 7))
plt.subplots_adjust(bottom=0.30)

initial_shift = 0
et_shifted = np.interp(t_bold - initial_shift, time_et, et_raw)

[line_bold] = ax.plot(t_bold, zscore(voxel_ts), linewidth=2, label=f"BOLD voxel (x = {nx}, y = {ny}, z = {nz})")
[line_co2] = ax.plot(t_bold, zscore(et_shifted),
                     linewidth=2, alpha=0.8,
                     label=f"EtCO₂ (shift = {initial_shift:.1f}s)")
ax.set_ylim(-3, 3)


ax.set_xlabel("Time (s)")
ax.set_ylabel("Normalized amplitude")
ax.set_title(f"BOLD vs EtCO₂ — Voxel index {initial_voxel}")
ax.grid()
ax.legend()


ax_vox = plt.axes([0.15, 0.17, 0.7, 0.03], facecolor="lightgray")
slider_vox = Slider(
    ax=ax_vox,
    label="Voxel index",
    valmin=0,
    valmax=V - 1,
    valinit=initial_voxel,
    valstep=1
)

ax_shift = plt.axes([0.15, 0.10, 0.7, 0.03], facecolor="lightgray")
slider_shift = Slider(
    ax=ax_shift,
    label="EtCO₂ shift (s)",
    valmin=-700,
    valmax=50,
    valinit=0,
    valstep=0.5
)

def update(_):
    voxel_idx = int(slider_vox.val)
    shift = slider_shift.val

    voxel_ts = bold_flat[voxel_idx, :]

    et_shifted = np.interp(t_bold - shift, time_et, et_raw)

    line_bold.set_ydata(zscore(voxel_ts))
    line_co2.set_ydata(zscore(et_shifted))

    ax.set_title(f"BOLD vs EtCO₂ — Voxel index {voxel_idx}")
    line_co2.set_label(f"EtCO₂ (shift = {shift:.1f}s)")
    ax.legend()

    fig.canvas.draw_idle()


slider_vox.on_changed(update)
slider_shift.on_changed(update)

plt.show(block=True)
