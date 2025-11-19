import os
from .process_video import process_single_video # Importamos la función del otro script

def main():
    """
    Encuentra y procesa todos los videos .mp4 en el directorio de datasets,
    y guarda los resultados en una estructura de carpetas paralela.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Directorio raíz de los videos de entrada
    root_input_dir = os.path.join(base_path, 'Dataset_Ejercicios')
    
    # Directorio raíz donde se guardarán los resultados
    root_output_dir = os.path.join(base_path, 'resultados_analisis')

    print("-" * 60)
    print(f"Iniciando procesamiento masivo de videos...")
    print(f"Directorio de entrada: {root_input_dir}")
    print(f"Directorio de salida:  {root_output_dir}")
    print("-" * 60)

    # Recorrer recursivamente el directorio de entrada
    for dirpath, _, filenames in os.walk(root_input_dir):
        for filename in filenames:
            if filename.lower().endswith('.mp4'):
                
                # Construir la ruta completa del video de entrada
                video_input_path = os.path.join(dirpath, filename)
                
                # Crear la estructura de carpetas de salida correspondiente
                relative_path = os.path.relpath(dirpath, root_input_dir)
                output_dir = os.path.join(root_output_dir, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                
                # Definir las rutas de los archivos de salida
                base_filename = os.path.splitext(filename)[0]
                video_output_path = os.path.join(output_dir, f"{base_filename}.mp4")
                csv_output_path = os.path.join(output_dir, f"{base_filename}.csv")

                # Llamar a la función de procesamiento para el video actual
                process_single_video(video_input_path, video_output_path, csv_output_path)

    print("-" * 60)
    print("Procesamiento masivo completado.")
    print("-" * 60)

if __name__ == '__main__':
    main()
