import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('sub-01_ses-01_gas_traces.txt', sep=r'\s+')

print(df.columns.tolist())
print(df.head())


if 'Time' in df.columns:
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')

plt.figure(figsize=(10, 6))
plt.plot(df['sec'], df['PctCO2'], label='End-tidal CO₂ (%)', color='red')
plt.plot(df['sec'], df['PctO2'], label='End-tidal O₂ (%)', color='blue')

plt.title('End-Tidal Gas Concentrations Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Partial Pressure (%)')
plt.legend()
plt.grid(True)
plt.show()
