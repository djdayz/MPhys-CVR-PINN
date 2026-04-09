import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np

df = pd.read_csv("/Users/mac/PycharmProjects/pythonMPhysproject/sub-01_ses-01_gas_traces.txt", sep='\t')

time = df["sec"]
co2_pct = df["PctCO2"]

pct_to_mmhg = 760.0 / 100.0
co2 = co2_pct * pct_to_mmhg


peaks, _ = find_peaks(co2, distance=60, prominence=0.005)
etco2_times = time.iloc[peaks]
etco2_values = co2.iloc[peaks]

etco2_interp = np.interp(time, etco2_times, etco2_values)

etco2_baseline = np.median(etco2_values)
baseline_line = np.ones_like(time) * etco2_baseline

print(f"EtCO₂ baseline = {etco2_baseline:.2f} mmHg")

df["EtCO2_interp_mmHg"] = etco2_interp
df["EtCO2_baseline_mmHg"] = etco2_baseline

output_path = "bids_dir/sub-01/ses-01/pre/EtCO2_mmHg.txt"
df.to_csv(output_path, sep='\t', index=False, float_format='%.5f')

print(f"Detected {len(etco2_values)} EtCO₂ peaks. Mean EtCO₂ = {etco2_values.mean():.3f}%")

plt.figure(figsize=(12, 6))
plt.plot(time, co2, color='lightsteelblue', linewidth=1)
plt.plot(etco2_times, etco2_values, color = "blue", linewidth=2, label='Detected EtCO₂')

plt.xlabel("Time (sec)")
plt.ylabel("CO₂ (mmHg)")
plt.title("End-tidal CO₂ (EtCO₂) Trend")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
