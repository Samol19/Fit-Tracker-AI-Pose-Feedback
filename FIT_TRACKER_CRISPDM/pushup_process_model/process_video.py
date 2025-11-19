import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Funciones de cálculo
def calculate_angle(a, b, c):
    """Calcula el ángulo entre tres puntos."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

def process_single_video(video_input_path, video_output_path, csv_output_path):
    """
    Procesa un único video para extraer landmarks, calcular ángulos y la nueva
    métrica de alineación de cadera, y guardar los datos en un CSV.
    """
    print(f"Procesando: {os.path.basename(video_input_path)}")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_input_path}")
        return

    # Configuración del video de salida
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

    data = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            
            # Extracción de puntos clave
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Cálculo de métricas (mismo sistema que plank)
            body_angle = calculate_angle(shoulder_l, hip_l, ankle_l)
            hip_shoulder_vertical_diff = hip_l[1] - shoulder_l[1]
            hip_ankle_vertical_diff = hip_l[1] - ankle_l[1]
            shoulder_elbow_angle = calculate_angle(hip_l, shoulder_l, elbow_l)
            wrist_shoulder_hip_angle = calculate_angle(wrist_l, shoulder_l, hip_l)
            shoulder_wrist_vertical_diff = shoulder_l[1] - wrist_l[1]
            
            # Métrica clave: ángulo del codo
            # ~180° = brazos extendidos (arriba)
            # ~90° o menos = brazos flexionados (abajo)
            elbow_angle = calculate_angle(shoulder_l, elbow_l, wrist_l)

            # Guardar datos
            data.append({
                'frame': frame_idx,
                'body_angle': body_angle,
                'hip_shoulder_vertical_diff': hip_shoulder_vertical_diff,
                'hip_ankle_vertical_diff': hip_ankle_vertical_diff,
                'shoulder_elbow_angle': shoulder_elbow_angle,
                'wrist_shoulder_hip_angle': wrist_shoulder_hip_angle,
                'shoulder_wrist_vertical_diff': shoulder_wrist_vertical_diff,
                'elbow_angle': elbow_angle,  # MÉTRICA CLAVE PARA REPETICIONES
            })

            # Visualización
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            
            # Muestra los datos en pantalla
            cv2.putText(image, f"Body Angle: {int(body_angle)}", tuple(np.multiply(hip_l, [frame_width, frame_height]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            pass

        out.write(image)
        frame_idx += 1

    # Guardado y limpieza
    cap.release()
    out.release()
    pose.close()

    # Guardar los datos en un archivo CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_output_path, index=False)
    print(f"Resultados guardados en: {os.path.basename(csv_output_path)}")

if __name__ == '__main__':
    # Ejemplo de uso
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    video_in = os.path.join(base_dir, 'Dataset_Ejercicios', 'pushup', 'pushup_correcto', 'IMG_0197.mov') # Ajustado a un nombre de archivo real
    video_out_dir = os.path.join(base_dir, 'resultados_analisis', 'pushup', 'pushup_correcto')
    os.makedirs(video_out_dir, exist_ok=True)
    video_out = os.path.join(video_out_dir, 'IMG_0197_procesado.mp4')
    csv_out = os.path.join(video_out_dir, 'IMG_0197_datos.csv')

    if not os.path.exists(video_in):
        print("\n--- ADVERTENCIA ---")
        print(f"El video de ejemplo no se encontró en: {video_in}")
        print("El script se ha creado, pero para probarlo directamente, necesitarás ajustar la ruta.")
    else:
        process_single_video(video_in, video_out, csv_out)