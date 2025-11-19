import pandas as pd
import matplotlib.pyplot as plt

data_path = "FIT_TRACKER/resultados_analisis/pushup/pushup_cadera_caida/IMG_0418_datos.csv"

df = pd.read_csv(data_path)

features = [
    'body_angle',
    'hip_shoulder_vertical_diff',
    'hip_ankle_vertical_diff',
    'shoulder_elbow_angle',
    'wrist_shoulder_hip_angle'
]

plt.figure(figsize=(15, 10))
for i, col in enumerate(features, 1):
    plt.subplot(len(features) + 1, 1, i)
    plt.plot(df['frame'], df[col], label=col)
    plt.title(col)
    plt.grid(True)
    plt.legend()

from scipy.signal import find_peaks, savgol_filter

window_length = 31
polyorder = 3
distance = 40
prominence = 5

signal = df['hip_ankle_vertical_diff']
smoothed_signal = savgol_filter(signal, window_length=window_length, polyorder=polyorder)

peaks, _ = find_peaks(smoothed_signal, distance=distance, prominence=prominence)
valleys, _ = find_peaks(-smoothed_signal, distance=distance, prominence=prominence)

plt.subplot(len(features) + 1, 1, len(features) + 1)
plt.plot(df['frame'], signal, label='hip_ankle_vertical_diff Original', alpha=0.4)
plt.plot(df['frame'], smoothed_signal, label='hip_ankle_vertical_diff Suavizado', color='orange', linewidth=2)
plt.scatter(df['frame'].iloc[peaks], smoothed_signal[peaks], color='green', marker='^', s=100, label='Picos')
plt.scatter(df['frame'].iloc[valleys], smoothed_signal[valleys], color='red', marker='v', s=100, label='Valles')
plt.title('hip_ankle_vertical_diff (original, suavizado, picos y valles)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
