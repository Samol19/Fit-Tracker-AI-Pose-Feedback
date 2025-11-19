import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def visualize_squat_data(csv_path, png_output_path):
    """Carga datos de análisis de squat desde CSV y guarda un gráfico como PNG."""
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f'Squat Analysis: {os.path.basename(csv_path)}', fontsize=16)

    axs[0].plot(df['frame'], df['left_knee_angle'], label='Left Knee Angle')
    axs[0].plot(df['frame'], df['right_knee_angle'], label='Right Knee Angle')
    axs[0].set_ylabel('Angle (degrees)')
    axs[0].set_title('Knee Angles (Depth Indicator)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(df['frame'], df['left_hip_angle'], label='Left Hip Angle')
    axs[1].plot(df['frame'], df['right_hip_angle'], label='Right Hip Angle')
    axs[1].set_ylabel('Angle (degrees)')
    axs[1].set_title('Hip Angles (Depth Indicator)')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(df['frame'], df['knee_distance'], label='Knee Distance', color='g')
    axs[2].set_ylabel('Normalized Distance')
    axs[2].set_title('Distance Between Knees (Knee Valgus Indicator)')
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(df['frame'], df['hip_shoulder_distance'], label='Hip-Shoulder Distance', color='r')
    axs[3].set_xlabel('Frame')
    axs[3].set_ylabel('Normalized Distance')
    axs[3].set_title('Horizontal Hip-Shoulder Distance (Torso Lean Indicator)')
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    plt.savefig(png_output_path)
    plt.close(fig)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        csv_file_path = sys.argv[1]
        output_dir = os.path.join(os.path.dirname(csv_file_path), 'test_visuals')
        os.makedirs(output_dir, exist_ok=True)
        file_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        png_file_path = os.path.join(output_dir, f"{file_name}.png")
    else:
        csv_file_path = r'\FIT_TRACKER\resultados_analisis\squat\squat_valgo_rodilla\squat_valgo_rodilla_front\valgo_rodilla_1_1_front.csv'
        output_dir = r'\FIT_TRACKER\test_visuals'
        os.makedirs(output_dir, exist_ok=True)
        png_file_path = os.path.join(output_dir, 'default_visualization.png')
        print(f"No file path provided. Using default: {csv_file_path}")

    print(f"Generating plot for {csv_file_path}...")
    visualize_squat_data(csv_file_path, png_file_path)
    print(f"Plot saved to {png_file_path}")
