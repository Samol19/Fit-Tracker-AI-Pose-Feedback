import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import joblib

SENSIBILIDAD_CLASE = {
    'plank_cadera_caida': 1.0,
    'plank_codos_abiertos': 0.5,
    'plank_correcto': 1.3, 
    'plank_pelvis_levantada': 1.0,
}

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        model = joblib.load(os.path.join(script_dir, 'plank_classifier_model.pkl'))
        le = joblib.load(os.path.join(script_dir, 'plank_label_encoder.pkl'))
        print("Modelo y codificador cargados.")
    except FileNotFoundError:
        print("Error: Archivos del modelo no encontrados. Ejecuta 'train_model.py'.")
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    BUFFER_SIZE_SECONDS = 1
    FPS_ESTIMADO = 30 
    BUFFER_FRAME_SIZE = BUFFER_SIZE_SECONDS * FPS_ESTIMADO
    feature_buffer = [] 
    current_prediction = "Iniciando..."
    prediction_proba_display = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
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
            
            current_features = [body_angle, hip_shoulder_vertical_diff, hip_ankle_vertical_diff, shoulder_elbow_angle, wrist_shoulder_hip_angle]
            feature_buffer.append(current_features)

            if len(feature_buffer) >= BUFFER_FRAME_SIZE:
                df_buffer = pd.DataFrame(feature_buffer, columns=[
                    'body_angle', 'hip_shoulder_vertical_diff', 'hip_ankle_vertical_diff',
                    'shoulder_elbow_angle', 'wrist_shoulder_hip_angle'
                ])
                features_for_prediction = []
                for col in df_buffer.columns:
                    features_for_prediction.extend([
                        df_buffer[col].mean(), df_buffer[col].std(), df_buffer[col].min(),
                        df_buffer[col].max(), df_buffer[col].max() - df_buffer[col].min()
                    ])
                
                prediction_proba = model.predict_proba([features_for_prediction])[0]
                print(prediction_proba)
                adjusted_probas = {clase: prob * SENSIBILIDAD_CLASE.get(clase, 1.0) for clase, prob in zip(le.classes_, prediction_proba)}
                total_adjusted_proba = sum(adjusted_probas.values())
                final_probas = {clase: prob / total_adjusted_proba for clase, prob in adjusted_probas.items()} if total_adjusted_proba > 0 else adjusted_probas
                current_prediction = max(final_probas, key=final_probas.get)
                prediction_proba_display = dict(sorted(final_probas.items(), key=lambda item: item[1], reverse=True))
                feature_buffer.clear()

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        except Exception as e:
            feature_buffer.clear()
            current_prediction = "No se detecta cuerpo"
            prediction_proba_display = {}

        # Interfaz de Feedback...
        h, w, _ = image.shape
        feedback_x = int(w * 0.02)
        feedback_y = int(h * 0.05)
        cv2.rectangle(image, (feedback_x - 10, feedback_y - 30), (feedback_x + 450, feedback_y + 150), (50, 50, 50, 200), -1)
        cv2.putText(image, "ESTADO:", (feedback_x, feedback_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, current_prediction, (feedback_x + 120, feedback_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2, cv2.LINE_AA)
        cv2.putText(image, "CONFIDENCIA:", (feedback_x, feedback_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y_pos = feedback_y + 70
        if prediction_proba_display:
            top_prediction_prob = list(prediction_proba_display.values())[0]
            cv2.putText(image, f"{top_prediction_prob:.0%}", (feedback_x + 190, feedback_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            for i, (clase, prob) in enumerate(prediction_proba_display.items()):
                if i < 4:
                    text = f"- {clase.replace('plank_', '')}: {prob:.0%}"
                    cv2.putText(image, text, (feedback_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                    y_pos += 20
        cv2.imshow('Feedback de Plancha en Tiempo Real', image)
        if cv2.waitKey(5) & 0xFF == 27: break
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == '__main__':
    main()