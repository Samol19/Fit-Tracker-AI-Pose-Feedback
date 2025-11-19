import os
import pandas as pd
import numpy as np

def find_stable_segments(df, num_segments=3, window_size_sec=2, fps=30):
    window_size_frames = int(window_size_sec * fps)
    if len(df) < window_size_frames:
        return [df] if len(df) > 0 else []
    stabilities = []
    for i in range(len(df) - window_size_frames + 1):
        window = df.iloc[i:i + window_size_frames]
        stability_score = window['body_angle'].std() + window['hip_shoulder_vertical_diff'].std()
        stabilities.append((stability_score, window))
    stabilities.sort(key=lambda x: x[0])
    return [window for score, window in stabilities[:num_segments] if not window.empty]

def build_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(script_dir)
    data_source_dir = os.path.join(base_path, 'resultados_analisis', 'plank')
    output_csv_path = os.path.join(base_path, 'plank_training_dataset.csv')

    if not os.path.exists(data_source_dir):
        print(f"Error: Directorio '{data_source_dir}' no encontrado.")
        return

    all_features = []
    # Columnas de características
    feature_columns = [
        'body_angle', 
        'hip_shoulder_vertical_diff', 
        'hip_ankle_vertical_diff',
        'shoulder_elbow_angle',
        'wrist_shoulder_hip_angle'
    ]
    
    classes_to_ignore = ['plank_cabeza-levantada']
    print(f"Ignorando clases: {classes_to_ignore}")
    print("Construyendo dataset...")

    for class_name in os.listdir(data_source_dir):
        class_dir = os.path.join(data_source_dir, class_name)
        if not os.path.isdir(class_dir) or class_name in classes_to_ignore:
            if os.path.isdir(class_dir): print(f"  -> Ignorando clase: {class_name}")
            continue
        print(f"  Procesando clase: {class_name}")
        for filename in os.listdir(class_dir):
            if filename.endswith('_datos.csv'):
                csv_path = os.path.join(class_dir, filename)
                try:
                    df = pd.read_csv(csv_path)
                    if df.empty: continue
                    stable_segments = find_stable_segments(df, num_segments=3)
                    if not stable_segments: continue
                    for i, segment_df in enumerate(stable_segments):
                        video_features = {'class': class_name, 'video_segment': f"{os.path.splitext(filename)[0]}_seg_{i+1}"}
                        for col in feature_columns:
                            video_features[f'{col}_mean'] = segment_df[col].mean()
                            video_features[f'{col}_std'] = segment_df[col].std()
                            video_features[f'{col}_min'] = segment_df[col].min()
                            video_features[f'{col}_max'] = segment_df[col].max()
                            video_features[f'{col}_range'] = segment_df[col].max() - segment_df[col].min()
                        all_features.append(video_features)
                except Exception as e:
                    print(f"    - Error en {filename}: {e}")

    if not all_features:
        print("\nNo se extrajeron características.")
        return
        
    final_df = pd.DataFrame(all_features)
    final_df.to_csv(output_csv_path, index=False)
    print("-" * 50)
    print(f"Dataset creado en '{os.path.basename(output_csv_path)}'!")
    print(f"Total de muestras: {len(final_df)}")
    print("Clases en el dataset:")
    print(final_df['class'].value_counts())
    print("-" * 50)

if __name__ == '__main__':
    build_dataset()