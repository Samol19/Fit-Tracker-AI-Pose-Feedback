import os
from process_video import process_single_video

def main():
    """Procesa todos los videos .mov en el directorio de datasets de pushup."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(script_dir)
    
    # Directorio raíz de los videos de entrada (específico para Push-up)
    root_input_dir = os.path.join(base_path, 'Dataset_Ejercicios', 'pushup')
    
    # Directorio raíz donde se guardarán los resultados
    root_output_dir = os.path.join(base_path, 'resultados_analisis', 'pushup')

    print("Iniciando procesamiento masivo de videos de flexiones...")
    print(f"Directorio de entrada: {root_input_dir}")
    print(f"Directorio de salida:  {root_output_dir}")

    # Verificar si el directorio de entrada existe
    if not os.path.exists(root_input_dir):
        print(f"¡ERROR! El directorio de entrada no existe: {root_input_dir}")
        print("Asegúrate de tener una carpeta 'pushup' dentro de 'Dataset_Ejercicios'.")
        return

    # Recorrer recursivamente el directorio de entrada
    for dirpath, _, filenames in os.walk(root_input_dir):
        for filename in filenames:
            if filename.lower().endswith('.mov'):
                
                # Construir la ruta completa del video de entrada
                video_input_path = os.path.join(dirpath, filename)
                
                # Crear la estructura de carpetas de salida correspondiente
                relative_path = os.path.relpath(dirpath, root_input_dir)
                output_dir = os.path.join(root_output_dir, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                
                # Definir las rutas de los archivos de salida
                base_filename = os.path.splitext(filename)[0]
                # Guardaremos los videos procesados como .mp4 por compatibilidad
                video_output_path = os.path.join(output_dir, f"{base_filename}_procesado.mp4")
                csv_output_path = os.path.join(output_dir, f"{base_filename}_datos.csv")

                # Llamar a la función de procesamiento para el video actual
                try:
                    process_single_video(video_input_path, video_output_path, csv_output_path)
                except Exception as e:
                    print(f"Error al procesar {filename}: {e}")
                    continue

    print("Procesamiento masivo de flexiones completado.")

if __name__ == '__main__':
    main()