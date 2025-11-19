import pandas as pd
import numpy as np
import os
from detect_repetitions import detect_squat_repetitions, smooth_signal

def build_dataset(root_dir):
    """Construye un dataset de features a partir de los CSV de análisis."""
    feature_list = []

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(subdir, file)
                
                # Extraer etiqueta del path
                # El label es el nombre del directorio padre
                try:
                    label = os.path.basename(os.path.dirname(subdir))
                    if label.endswith(('_front', '_diag', '_lat')):
                         label = os.path.basename(os.path.dirname(os.path.dirname(subdir)))

                except Exception:
                    label = 'unknown'

                # Detectar repeticiones
                # Ejecutar sin visualización para procesamiento en lote
                repetition_frames = detect_squat_repetitions(csv_path, visualize=False)
                
                if not repetition_frames.any():
                    continue

                df = pd.read_csv(csv_path)

                # Extraer features para cada repetición (con estadísticas)
                for frame_idx in repetition_frames:
                    # Definir ventana alrededor del punto más profundo (±15 frames)
                    window_start = max(0, frame_idx - 15)
                    window_end = min(len(df), frame_idx + 15)
                    window_df = df.iloc[window_start:window_end]
                    
                    # Calcular features base para la ventana
                    window_df['avg_knee_angle'] = window_df[['left_knee_angle', 'right_knee_angle']].mean(axis=1)
                    window_df['avg_hip_angle'] = window_df[['left_hip_angle', 'right_hip_angle']].mean(axis=1)
                    
                    # Feature columns a procesar
                    feature_columns = ['avg_knee_angle', 'avg_hip_angle', 'knee_distance', 'hip_shoulder_distance']
                    
                    features = {
                        'label': label,
                        'source_file': file,
                        'frame': frame_idx
                    }
                    
                    # Calcular 5 estadísticas para cada feature
                    for col in feature_columns:
                        if col in window_df.columns and not window_df[col].dropna().empty:
                            series = window_df[col]
                            features[f'{col}_mean'] = series.mean()
                            features[f'{col}_std'] = series.std()
                            features[f'{col}_min'] = series.min()
                            features[f'{col}_max'] = series.max()
                            features[f'{col}_range'] = series.max() - series.min()
                        else:
                            # Valores por defecto si falta la columna
                            features[f'{col}_mean'] = 0
                            features[f'{col}_std'] = 0
                            features[f'{col}_min'] = 0
                            features[f'{col}_max'] = 0
                            features[f'{col}_range'] = 0
                    
                    feature_list.append(features)

    return pd.DataFrame(feature_list)

if __name__ == '__main__':
    # Directory containing the per-frame analysis CSVs
    analysis_dir = r'\FIT_TRACKER\resultados_analisis\squat'
    
    # Path for the final output dataset
    output_csv_path = r'\FIT_TRACKER\squat_training_dataset.csv'

    print("Building dataset from analysis files...")
    final_dataset = build_dataset(analysis_dir)
    
    if not final_dataset.empty:
        # Save the dataset to a new CSV file
        final_dataset.to_csv(output_csv_path, index=False)
        print(f"\nDataset successfully created!")
        print(f"Total repetitions found: {len(final_dataset)}")
        print(f"Dataset saved to: {output_csv_path}")
        print("\nFirst 5 rows of the dataset:")
        print(final_dataset.head())
        print("\nClass distribution:")
        print(final_dataset['label'].value_counts())
    else:
        print("No repetitions were found. The dataset is empty.")
