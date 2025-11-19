import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
import os

def detect_pushup_repetitions(csv_path):
    """Detección usando shoulder_wrist_vertical_diff.
    
    Valores más negativos = brazos extendidos
    Valores menos negativos (picos) = brazos flexionados
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {csv_path}")
        return

    if len(df) < 50:
        print(f"Video {os.path.basename(csv_path)} demasiado corto, omitiendo.")
        return

    if 'shoulder_wrist_vertical_diff' not in df.columns:
        print(f"❌ Columna 'shoulder_wrist_vertical_diff' no encontrada.")
        return
    
    signal = df['shoulder_wrist_vertical_diff'].values
    
    # Suavizado con Savitzky-Golay
    if len(signal) >= 11:
        smoothed_signal = savgol_filter(signal, window_length=11, polyorder=3)
    else:
        smoothed_signal = signal.copy()
    
    # Detección por picos (máximos)
    # Valores menos negativos (picos) = brazos flexionados = punto más bajo
    
    # Calcular estadísticas
    signal_range = smoothed_signal.max() - smoothed_signal.min()
    signal_min = smoothed_signal.min()
    signal_max = smoothed_signal.max()
    
    # Parámetros adaptativos MUY PERMISIVOS para capturar las 3 repeticiones
    # height: Debe estar en la mitad superior del rango (valores menos negativos)
    height_threshold = signal_min + (signal_range * 0.40)  # 40% del rango desde abajo
    distance_min = 25  # Mínimo 25 frames entre repeticiones
    prominence_min = max(signal_range * 0.10, 0.03)  # Mínimo 10% del rango O 0.03 absoluto
    
    # Detectar PICOS (máximos) = brazos flexionados
    peaks, properties = find_peaks(
        smoothed_signal,
        height=height_threshold,
        distance=distance_min,
        prominence=prominence_min
    )
    
    print(f"  Señal shoulder_wrist_vertical_diff: min={signal_min:.3f}, max={signal_max:.3f}, rango={signal_range:.3f}")
    print(f"  Parámetros: valor>{height_threshold:.3f}, distance>{distance_min}, prominence>{prominence_min:.3f}")
    print(f"  Detectados {len(peaks)} picos (repeticiones)")
    
    if len(peaks) == 0:
        print(f"\nNo se detectaron repeticiones en {os.path.basename(csv_path)}")
        return
    
    # Extracción y guardado
    output_dir = os.path.dirname(csv_path)
    base_filename = os.path.splitext(os.path.basename(csv_path))[0]

    # Limpiar archivos previos
    for item in os.listdir(output_dir):
        if item.startswith(base_filename) and "_rep_" in item:
            os.remove(os.path.join(output_dir, item))

    print(f"Detectadas {len(peaks)} repeticiones en {base_filename}")

    # Extraer ventanas alrededor de cada pico
    margin_before = 20
    margin_after = 20
    n_frames = len(df)
    rep_ranges = []
    
    for i, peak_idx in enumerate(peaks):
        start = max(0, peak_idx - margin_before)
        end = min(n_frames, peak_idx + margin_after)
        
        # Guardar repetición
        rep_df = df.iloc[start:end].copy()
        rep_filename = f"{base_filename}_rep_{i+1}.csv"
        rep_output_path = os.path.join(output_dir, rep_filename)
        rep_df.to_csv(rep_output_path, index=False)
        rep_ranges.append((start, end, peak_idx))
        
        rep_signal = smoothed_signal[start:end]
        movement_range = rep_signal.max() - rep_signal.min()
        max_value = smoothed_signal[peak_idx]
        print(f"  Rep {i+1}: frames {start}-{end}, valor máx={max_value:.3f}, rango={movement_range:.3f}")

    # Visualización
    analisis_dir = os.path.join(os.path.dirname(output_dir), 'analisis')
    os.makedirs(analisis_dir, exist_ok=True)
    
    fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f'Detección por shoulder_wrist_vertical_diff - {base_filename}', fontsize=14, fontweight='bold')
    
    # Subplot 1: Señal con picos detectados
    axs[0].plot(df.index, signal, label='Original', alpha=0.4, color='lightblue')
    axs[0].plot(df.index, smoothed_signal, label='Suavizado', color='darkblue', linewidth=2)
    axs[0].plot(peaks, smoothed_signal[peaks], "^", color='red', markersize=15, 
               label=f'Picos detectados ({len(peaks)})', zorder=5)
    
    # Marcar repeticiones
    for idx, (start, end, peak) in enumerate(rep_ranges):
        axs[0].axvspan(start, end, color='yellow', alpha=0.15)
    
    axs[0].axhline(y=height_threshold, color='orange', linestyle='--', linewidth=1.5, 
                   label=f'Umbral ({height_threshold:.3f})', alpha=0.7)
    axs[0].axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    
    axs[0].set_ylabel('Shoulder-Wrist Vertical Diff', fontsize=11)
    axs[0].set_title('Detección por Picos: Mayor valor (menos negativo) = Brazos flexionados (abajo)', fontsize=12)
    axs[0].legend(loc='upper right', fontsize=9)
    axs[0].grid(True, alpha=0.3)
    
    # Subplot 2: Comparación con elbow_angle si existe
    if 'elbow_angle' in df.columns:
        elbow_signal = df['elbow_angle']
        smoothed_elbow = savgol_filter(elbow_signal, window_length=11, polyorder=3) if len(elbow_signal) >= 11 else elbow_signal
        
        # Normalizar ambas señales para comparación visual
        norm_shoulder = (smoothed_signal - signal_min) / signal_range if signal_range > 0 else smoothed_signal
        elbow_range = smoothed_elbow.max() - smoothed_elbow.min()
        norm_elbow = (smoothed_elbow - smoothed_elbow.min()) / elbow_range if elbow_range > 0 else smoothed_elbow
        
        axs[1].plot(df.index, norm_elbow, label='elbow_angle (normalizado)', 
                    color='green', linewidth=2, alpha=0.7, linestyle='--')
        axs[1].plot(df.index, norm_shoulder, label='shoulder_wrist_vertical_diff (normalizado)', 
                    color='darkblue', linewidth=2, alpha=0.8)
        
        for peak in peaks:
            axs[1].axvline(x=peak, color='red', linestyle=':', alpha=0.4)
        
        axs[1].set_xlabel('Frame', fontsize=11)
        axs[1].set_ylabel('Valor Normalizado (0-1)', fontsize=11)
        axs[1].set_title('Comparación: shoulder_wrist_vertical_diff vs elbow_angle', fontsize=12)
        axs[1].legend(loc='upper right')
        axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plot_filename = f"{base_filename}_analisis.png"
    plot_output_path = os.path.join(analisis_dir, plot_filename)
    plt.savefig(plot_output_path, dpi=100)
    plt.close()
    print(f"\n📊 Gráfico guardado: {plot_filename}")


if __name__ == '__main__':
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(script_dir)
        example_csv = os.path.join(base_dir, 'resultados_analisis', 'pushup', 'pushup_codos_abiertos', 'IMG_0205_datos.csv')
        print(f"Ejecutando detección por shoulder_wrist_vertical_diff:\n{example_csv}\n")
        detect_pushup_repetitions(example_csv)
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"Ocurrió un error: {e}")
