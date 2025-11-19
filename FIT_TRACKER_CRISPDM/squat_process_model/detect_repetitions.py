import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys
import os

def smooth_signal(data, window_size=4):
    """Aplica un promedio móvil para suavizar una señal."""
    return data.rolling(window=window_size, min_periods=1, center=True).mean()

def detect_squat_repetitions(csv_path, visualize=True):
    """Detecta repeticiones de squat y visualiza señales originales vs suavizadas."""
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return []

    df = pd.read_csv(csv_path)

    knee_angle_signal = df['left_knee_angle']
    smoothed_knee_angle = smooth_signal(knee_angle_signal)
    
    peaks, _ = find_peaks(-smoothed_knee_angle, height=-160, distance=15)

    if visualize:
        fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        fig.suptitle(f'Signal Analysis: {os.path.basename(csv_path)}', fontsize=16)

        axs[0].plot(df['frame'], knee_angle_signal, label='Original Knee Angle', alpha=0.4)
        axs[0].plot(df['frame'], smoothed_knee_angle, label='Smoothed Knee Angle', color='orange', linewidth=2)
        axs[0].plot(peaks, smoothed_knee_angle[peaks], "x", color='red', markersize=10, label=f'Detected Reps ({len(peaks)})')
        
        axs[0].set_ylabel('Angle (degrees)')
        axs[0].set_title('Repetition Detection based on Knee Angle')
        axs[0].legend()
        axs[0].grid(True)

        knee_dist_signal = df['knee_distance']
        smoothed_knee_dist = smooth_signal(knee_dist_signal)
        axs[1].plot(df['frame'], knee_dist_signal, label='Original Knee Distance', color='g', alpha=0.4)
        axs[1].plot(df['frame'], smoothed_knee_dist, label='Smoothed Knee Distance', color='darkgreen', linewidth=2)

        hip_shoulder_dist_signal = df['hip_shoulder_distance']
        smoothed_hip_shoulder_dist = smooth_signal(hip_shoulder_dist_signal)
        axs[1].plot(df['frame'], hip_shoulder_dist_signal, label='Original Hip-Shoulder Dist', color='b', alpha=0.4)
        axs[1].plot(df['frame'], smoothed_hip_shoulder_dist, label='Smoothed Hip-Shoulder Dist', color='darkblue', linewidth=2)
        
        for peak in peaks:
            axs[1].axvline(x=peak, color='r', linestyle='--', alpha=0.6)

        axs[1].set_xlabel('Frame')
        axs[1].set_ylabel('Normalized Distance')
        axs[1].set_title('Feature Signals (Original vs. Smoothed)')
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()

    return peaks

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = r'\FIT_TRACKER\resultados_analisis\squat\squat_valgo_rodilla\squat_valgo_rodilla_diag\squat_valgo_rodilla_1_1_diag.csv'
        print(f"No file path provided. Using default for demonstration: {file_path}")

    detected_peaks = detect_squat_repetitions(file_path)
    print(f"\nDetection complete.")
    print(f"Found {len(detected_peaks)} repetitions at frames: {detected_peaks}")