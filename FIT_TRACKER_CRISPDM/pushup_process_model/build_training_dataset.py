import os
import pandas as pd
import numpy as np

def calculate_features(df):
    """Calcula estadísticas (mean, std, min, max, range) para cada feature."""
    features = {}
    
    # Columnas de features (igual que plank)
    feature_columns = [
        'body_angle',
        'hip_shoulder_vertical_diff',
        'hip_ankle_vertical_diff',
        'shoulder_elbow_angle',
        'wrist_shoulder_hip_angle',
        'shoulder_wrist_vertical_diff'
    ]
    
    for col in feature_columns:
        if col in df and not df[col].dropna().empty:
            series = df[col]
            features[f'{col}_mean'] = series.mean()
            features[f'{col}_std'] = series.std()
            features[f'{col}_min'] = series.min()
            features[f'{col}_max'] = series.max()
            features[f'{col}_range'] = series.max() - series.min()
        else:
            features[f'{col}_mean'] = 0
            features[f'{col}_std'] = 0
            features[f'{col}_min'] = 0
            features[f'{col}_max'] = 0
            features[f'{col}_range'] = 0
            
    return features

def main():
    """
    Recorre los directorios de resultados, procesa cada archivo de repetición (_rep_X.csv),
    calcula las nuevas características y construye el archivo CSV de entrenamiento.
    """
    # El script se encuentra en FIT_TRACKER/pushup_model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(script_dir) # Sube a FIT_TRACKER
    
    # Directorio raíz donde se encuentran los archivos de repeticiones
    root_input_dir = os.path.join(base_path, 'resultados_analisis', 'pushup')
    
    # Ruta del archivo CSV de salida que contendrá el dataset de entrenamiento
    output_csv_path = os.path.join(base_path, 'pushup_training_dataset.csv')

    print("-" * 60)
    print(f"Construyendo dataset de entrenamiento con NUEVAS características...")
    print(f"Directorio de entrada: {root_input_dir}")
    print(f"Archivo de salida:     {output_csv_path}")
    print("-" * 60)

    if not os.path.exists(root_input_dir):
        print(f"¡ERROR! El directorio de resultados no existe: {root_input_dir}")
        print("Asegúrate de haber ejecutado 'run_bulk_processing.py' y 'run_repetition_detection.py' primero.")
        return

    all_features = []

    # Recorrer recursivamente el directorio de entrada
    for dirpath, _, filenames in os.walk(root_input_dir):
        for filename in filenames:
            # Buscamos solo los archivos de repeticiones individuales
            if filename.endswith('.csv') and '_rep_' in filename:
                
                # Extraer la clase del nombre de la carpeta
                class_name = os.path.basename(dirpath)
                
                # Ruta completa al archivo de la repetición
                rep_csv_path = os.path.join(dirpath, filename)
                
                try:
                    df_rep = pd.read_csv(rep_csv_path)
                    
                    # Calcular las características para esta repetición
                    features = calculate_features(df_rep)
                    
                    # Añadir la clase (etiqueta) y el archivo de origen
                    features['class'] = class_name
                    features['source_file'] = filename
                    
                    all_features.append(features)
                    
                except Exception as e:
                    print(f"!!! ERROR al procesar {filename}: {e}")
                    continue

    if not all_features:
        print("¡ADVERTENCIA! No se encontraron archivos de repeticiones para procesar.")
        return

    # Crear un DataFrame final con todas las características y guardarlo
    final_df = pd.DataFrame(all_features)
    
    # Reordenar columnas para que 'class' y 'source_file' estén al final
    cols = [c for c in final_df.columns if c not in ['class', 'source_file']] + ['class', 'source_file']
    final_df = final_df[cols]
    
    final_df.to_csv(output_csv_path, index=False)

    print("-" * 60)
    print(f"¡Éxito! Dataset de entrenamiento creado con {len(final_df)} ejemplos.")
    print(f"Guardado en: {output_csv_path}")
    print("-" * 60)
    print("\nResumen de las clases encontradas:")
    print(final_df['class'].value_counts())


if __name__ == '__main__':
    main()