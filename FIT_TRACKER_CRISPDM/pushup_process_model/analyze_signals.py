"""Script de análisis para visualizar señales de pushup y calibrar umbrales óptimos."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import os
import sys

def analyze_pushup_signals(csv_path):
    """Analiza y visualiza las señales principales de un video de pushup."""
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró el archivo {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    if len(df) < 50:
        print(f"Video {os.path.basename(csv_path)} demasiado corto, omitiendo.")
        return
    
    # Extraer señales principales
    shoulder_wrist_signal = df['shoulder_wrist_vertical_diff']
    hip_ankle_signal = df['hip_ankle_vertical_diff']
    elbow_angle_signal = df['shoulder_elbow_angle']
    
    # Suavizado con Savitzky-Golay
    if len(shoulder_wrist_signal) >= 11:
        smoothed_sw = savgol_filter(shoulder_wrist_signal, window_length=11, polyorder=3)
        smoothed_ha = savgol_filter(hip_ankle_signal, window_length=11, polyorder=3)
        smoothed_elbow = savgol_filter(elbow_angle_signal, window_length=11, polyorder=3)
    else:
        smoothed_sw = shoulder_wrist_signal.rolling(window=5, min_periods=1, center=True).mean()
        smoothed_ha = hip_ankle_signal.rolling(window=5, min_periods=1, center=True).mean()
        smoothed_elbow = elbow_angle_signal.rolling(window=5, min_periods=1, center=True).mean()
    
    # Detectar picos y valles en shoulder_wrist_vertical_diff
    # En pushup: valor alto = brazos extendidos (arriba), valor bajo = brazos flexionados (abajo)
    # Buscamos valles (brazos abajo) como marcadores de repetición completa
    peaks_up, _ = find_peaks(smoothed_sw, height=-0.05, distance=20, prominence=0.05)
    valleys_down, _ = find_peaks(-smoothed_sw, height=0.15, distance=20, prominence=0.05)
    
    # Calcular estadísticas
    print(f"\nAnálisis: {os.path.basename(csv_path)}")
    print(f"\nShoulder-Wrist Vertical Diff (señal principal):")
    print(f"  Mín:  {shoulder_wrist_signal.min():.4f} (brazos muy abajo)")
    print(f"  Máx:  {shoulder_wrist_signal.max():.4f} (brazos muy arriba)")
    print(f"  Mean: {shoulder_wrist_signal.mean():.4f}")
    print(f"  Std:  {shoulder_wrist_signal.std():.4f}")
    print(f"  Rango: {shoulder_wrist_signal.max() - shoulder_wrist_signal.min():.4f}")
    
    print(f"\nHip-Ankle Vertical Diff (señal antigua):")
    print(f"  Mín:  {hip_ankle_signal.min():.4f}")
    print(f"  Máx:  {hip_ankle_signal.max():.4f}")
    print(f"  Mean: {hip_ankle_signal.mean():.4f}")
    
    print(f"\nElbow Angle:")
    print(f"  Mín:  {elbow_angle_signal.min():.1f}° (brazo muy flexionado)")
    print(f"  Máx:  {elbow_angle_signal.max():.1f}° (brazo extendido)")
    
    print(f"\nDetección de repeticiones:")
    print(f"  Picos detectados (arriba): {len(peaks_up)}")
    print(f"  Valles detectados (abajo): {len(valleys_down)}")
    print(f"  Repeticiones estimadas: {min(len(peaks_up), len(valleys_down))}")
    
    # Crear visualización completa
    fig, axs = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'Análisis de Señales - {os.path.basename(csv_path)}', fontsize=16, fontweight='bold')
    
    # 1. Shoulder-Wrist Vertical Diff (LA MÁS IMPORTANTE)
    axs[0].plot(df.index, shoulder_wrist_signal, label='Original', alpha=0.4, color='blue')
    axs[0].plot(df.index, smoothed_sw, label='Suavizado', color='darkblue', linewidth=2)
    axs[0].plot(peaks_up, smoothed_sw[peaks_up], "^", color='green', markersize=12, 
                label=f'Posición ARRIBA ({len(peaks_up)})')
    axs[0].plot(valleys_down, smoothed_sw[valleys_down], "v", color='red', markersize=12, 
                label=f'Posición ABAJO ({len(valleys_down)})')
    axs[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Referencia (0)')
    axs[0].set_ylabel('Diferencia Vertical', fontsize=10)
    axs[0].set_title('Shoulder-Wrist Vertical Diff (SEÑAL PRINCIPAL PARA DETECCIÓN)', 
                     fontsize=11, fontweight='bold')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)
    
    # 2. Hip-Ankle Vertical Diff (comparación con señal antigua)
    axs[1].plot(df.index, hip_ankle_signal, label='Original', alpha=0.4, color='orange')
    axs[1].plot(df.index, smoothed_ha, label='Suavizado', color='darkorange', linewidth=2)
    axs[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axs[1].set_ylabel('Diferencia Vertical', fontsize=10)
    axs[1].set_title('Hip-Ankle Vertical Diff (señal antigua, menos confiable)', fontsize=11)
    axs[1].legend(loc='upper right')
    axs[1].grid(True, alpha=0.3)
    
    # 3. Elbow Angle (señal complementaria)
    axs[2].plot(df.index, elbow_angle_signal, label='Original', alpha=0.4, color='purple')
    axs[2].plot(df.index, smoothed_elbow, label='Suavizado', color='darkviolet', linewidth=2)
    axs[2].set_ylabel('Ángulo (grados)', fontsize=10)
    axs[2].set_title('Shoulder-Elbow Angle (señal complementaria)', fontsize=11)
    axs[2].legend(loc='upper right')
    axs[2].grid(True, alpha=0.3)
    
    # 4. Comparación directa: Shoulder-Wrist vs Hip-Ankle
    axs[3].plot(df.index, smoothed_sw, label='Shoulder-Wrist (MEJOR)', 
                color='darkblue', linewidth=2, alpha=0.8)
    axs[3].plot(df.index, smoothed_ha, label='Hip-Ankle (antigua)', 
                color='darkorange', linewidth=2, alpha=0.6, linestyle='--')
    # Marcar las repeticiones detectadas
    for valley in valleys_down:
        axs[3].axvline(x=valley, color='red', linestyle=':', alpha=0.4)
    axs[3].set_xlabel('Frame', fontsize=11)
    axs[3].set_ylabel('Valor Normalizado', fontsize=10)
    axs[3].set_title('Comparación: ¿Cuál señal es mejor para detectar repeticiones?', fontsize=11)
    axs[3].legend(loc='upper right')
    axs[3].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()
    
    print(f"\n{'='*80}\n")


def analyze_multiple_videos(base_dir, categories=['pushup_correcto', 'pushup_cadera_caida', 
                                                  'pushup_codos_abiertos', 'pushup_pelvis_levantada']):
    """Analiza múltiples videos de diferentes categorías para encontrar patrones."""
    print("\nAnálisis comparativo de múltiples videos")
    
    all_stats = []
    
    for category in categories:
        category_dir = os.path.join(base_dir, 'resultados_analisis', 'pushup', category)
        if not os.path.exists(category_dir):
            continue
        
        # Buscar el primer archivo _datos.csv en esta categoría
        for filename in os.listdir(category_dir):
            if filename.endswith('_datos.csv') and '_rep_' not in filename:
                csv_path = os.path.join(category_dir, filename)
                df = pd.read_csv(csv_path)
                
                if len(df) < 50:
                    continue
                
                sw_signal = df['shoulder_wrist_vertical_diff']
                stats = {
                    'category': category,
                    'file': filename,
                    'sw_min': sw_signal.min(),
                    'sw_max': sw_signal.max(),
                    'sw_mean': sw_signal.mean(),
                    'sw_std': sw_signal.std(),
                    'sw_range': sw_signal.max() - sw_signal.min()
                }
                all_stats.append(stats)
                break  # Solo analizar 1 video por categoría
    
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        print("\nEstadísticas de Shoulder-Wrist Vertical Diff por categoría:")
        print(stats_df.to_string(index=False))
        
        # Calcular umbrales sugeridos
        overall_mean = stats_df['sw_mean'].mean()
        overall_std = stats_df['sw_std'].mean()
        overall_min = stats_df['sw_min'].min()
        overall_max = stats_df['sw_max'].max()
        
        print(f"\nUmbrales sugeridos para detección:")
        print(f"Valor típico cuando brazos ARRIBA: {overall_max:.4f} a {overall_mean + overall_std:.4f}")
        print(f"Valor típico cuando brazos ABAJO:  {overall_mean - overall_std:.4f} a {overall_min:.4f}")
        print(f"\nUmbral sugerido para detectar ABAJO: < {overall_mean - 0.5 * overall_std:.4f}")
        print(f"Umbral sugerido para detectar ARRIBA: > {overall_mean + 0.3 * overall_std:.4f}")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    print("\nHerramienta de análisis de señales pushup")
    print("Este script ayuda a entender las señales y calibrar umbrales óptimos.")
    
    # Si se proporciona un archivo específico como argumento
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        analyze_pushup_signals(csv_path)
    else:
        # Analizar videos de ejemplo de cada categoría
        print("Modo: Análisis comparativo de múltiples categorías\n")
        analyze_multiple_videos(base_dir)
        
        # Después, analizar en detalle un video correcto como ejemplo
        example_video = os.path.join(base_dir, 'resultados_analisis', 'pushup', 
                                    'pushup_correcto', 'IMG_0198_datos.csv')
        if os.path.exists(example_video):
            print("\n\nAhora mostrando análisis detallado de un video de ejemplo...")
            print("Presiona cualquier tecla en la ventana del gráfico para continuar.\n")
            input("Presiona ENTER para continuar...")
            analyze_pushup_signals(example_video)
        else:
            print(f"\nNo se encontró el video de ejemplo en: {example_video}")
            print("Ejecuta primero 'run_bulk_processing.py' para procesar los videos.")
