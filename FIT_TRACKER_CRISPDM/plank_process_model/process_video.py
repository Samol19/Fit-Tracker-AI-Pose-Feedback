import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def process_single_video(video_input_path, video_output_path, csv_output_path):
    print(f"Procesando: {os.path.basename(video_input_path)}")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_input_path}")
        return

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
            
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            body_angle = calculate_angle(shoulder_l, hip_l, ankle_l)
            hip_shoulder_vertical_diff = hip_l[1] - shoulder_l[1]
            hip_ankle_vertical_diff = hip_l[1] - ankle_l[1]
            shoulder_elbow_angle = calculate_angle(hip_l, shoulder_l, elbow_l)
            wrist_shoulder_hip_angle = calculate_angle(wrist_l, shoulder_l, hip_l)
            
            data.append({
                'frame': frame_idx,
                'body_angle': body_angle,
                'hip_shoulder_vertical_diff': hip_shoulder_vertical_diff,
                'hip_ankle_vertical_diff': hip_ankle_vertical_diff,
                'shoulder_elbow_angle': shoulder_elbow_angle,
                'wrist_shoulder_hip_angle': wrist_shoulder_hip_angle,
            })

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        except Exception as e:
            pass

        out.write(image)
        frame_idx += 1

    cap.release()
    out.release()
    pose.close()
    df = pd.DataFrame(data)
    df.to_csv(csv_output_path, index=False)
    print(f"Resultados guardados en: {os.path.basename(csv_output_path)}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    video_name = 'plank_test.mov' 
    clase = 'plank_correcto' 
    
    video_in = os.path.join(base_dir, 'Dataset_Ejercicios', 'plank', clase, video_name)
    
    video_out_dir = os.path.join(base_dir, 'resultados_analisis', 'plank', clase)
    os.makedirs(video_out_dir, exist_ok=True)
    
    video_out = os.path.join(video_out_dir, f'{os.path.splitext(video_name)[0]}_procesado.mp4')
    csv_out = os.path.join(video_out_dir, f'{os.path.splitext(video_name)[0]}_datos.csv')

    if not os.path.exists(video_in):
        print("\n--- ADVERTENCIA ---")
        print(f"El video de ejemplo no se encontró en: {video_in}")
        print("El script se ha creado, pero para probarlo directamente, necesitarás:")
        print("1. Crear la carpeta 'Dataset_Ejercicios/plank/plank_correcto'.")
        print("2. Poner un video de plancha allí.")
        print("3. Actualizar la variable 'video_name' en este script con el nombre de tu video.")
    else:
        process_single_video(video_in, video_out, csv_out)
