import os
from detect_repetitions import detect_pushup_repetitions

def main():
    """Procesa todos los archivos _datos.csv para detectar y segmentar repeticiones."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(script_dir)
    
    # Directorio raíz donde se encuentran los archivos CSV a procesar
    root_input_dir = os.path.join(base_path, 'resultados_analisis', 'pushup')

    print(f"Iniciando detección de repeticiones en: {root_input_dir}")

    # Verificar si el directorio de entrada existe
    if not os.path.exists(root_input_dir):
        print(f"¡ERROR! El directorio de resultados no existe: {root_input_dir}")
        print("Asegúrate de haber ejecutado 'run_bulk_processing.py' primero.")
        return

    # Recorrer recursivamente el directorio de entrada
    for dirpath, _, filenames in os.walk(root_input_dir):
        for filename in filenames:
            # Buscamos los archivos de datos generados en el paso anterior
            if filename.endswith('_datos.csv'):
                
                csv_input_path = os.path.join(dirpath, filename)
                
                # Llamar a la función de detección para el archivo actual
                try:
                    print(f"--- Procesando archivo: {filename} ---")
                    detect_pushup_repetitions(csv_input_path)
                except Exception as e:
                    print(f"Error al procesar {filename}: {e}")
                    continue

    print("Detección de repeticiones completada.")

if __name__ == '__main__':
    main()
