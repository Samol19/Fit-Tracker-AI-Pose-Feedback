import os
from visualize_data import visualize_squat_data

def run_bulk_visualization(root_dir, output_dir):
    """Recorre el directorio raíz, encuentra todos los archivos CSV y genera visualizaciones."""
    print(f"Starting bulk visualization...")
    print(f"Input directory: {root_dir}")
    print(f"Output directory: {output_dir}")

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(subdir, file)
                
                relative_path = os.path.relpath(subdir, root_dir)
                current_output_dir = os.path.join(output_dir, relative_path)
                os.makedirs(current_output_dir, exist_ok=True)
                
                png_filename = os.path.splitext(file)[0] + '.png'
                png_output_path = os.path.join(current_output_dir, png_filename)
                
                print(f"  - Processing: {csv_path}")
                try:
                    visualize_squat_data(csv_path, png_output_path)
                    print(f"    -> Saved plot to {png_output_path}")
                except Exception as e:
                    print(f"    -> Failed to process {csv_path}. Error: {e}")

    print("Bulk visualization complete.")

if __name__ == '__main__':
    analysis_results_dir = r'\FIT_TRACKER\resultados_analisis'
    
    visualizations_output_dir = r'\FIT_TRACKER\visualizaciones'
    
    os.makedirs(visualizations_output_dir, exist_ok=True)
    
    run_bulk_visualization(analysis_results_dir, visualizations_output_dir)
