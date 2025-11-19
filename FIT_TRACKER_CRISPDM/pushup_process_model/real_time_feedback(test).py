import warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import os
from scipy.signal import find_peaks, savgol_filter
from collections import deque

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

SENSIBILIDAD_CLASE = {
    "pushup_correcto": 1.0,
    "pushup_cadera_caida": 1.0,
    "pushup_codos_abiertos": 0.9,
    "pushup_pelvis_levantada": 1.0,
}

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        model = joblib.load(os.path.join(script_dir, 'pushup_classifier_model.pkl'))
        le = joblib.load(os.path.join(script_dir, 'pushup_label_encoder.pkl'))
        scaler = joblib.load(os.path.join(script_dir, 'pushup_scaler.pkl'))
        print("Modelo, codificador y scaler cargados correctamente.")
    except FileNotFoundError:
        print("\nNo se encontraron los archivos del modelo, codificador o scaler.")
        print("Asegúrate de haber ejecutado 'train_model.py' primero.")
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    rep_counter = 0
    feedback = 'Listo para empezar'
    last_rep_feedback = 'Ninguna'
    last_probabilities = np.zeros(len(le.classes_))
    avg_elbow_angle_display = 0
    avg_hip_ankle_display = 0
    last_shoulder_wrist_vertical_diff = 'N/A'

    print("Starting real-time pushup feedback. Press 'q' to quit.")
    debug_features = None
    debug_probabilities = None
    
    BUFFER_SIZE = 150
    signal_buffer = deque(maxlen=BUFFER_SIZE)
    features_buffer = deque(maxlen=BUFFER_SIZE)
    
    detected_peaks = []
    last_peak_frame = -50
    
    PEAK_MIN_DISTANCE = 25
    MARGIN_BEFORE = 20
    MARGIN_AFTER = 20
    MIN_PROMINENCE = 0.03
    
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h_cam, w_cam, _ = frame.shape
        dashboard_width = 400
        dashboard = np.zeros((h_cam, dashboard_width, 3), dtype="uint8")
        dashboard[:] = (41, 41, 41)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        current_feedback = "Buscando cuerpo..."
        try:
            landmarks = results.pose_landmarks.landmark
            def valid_landmark(lm):
                return lm is not None and 0 <= lm.x <= 1 and 0 <= lm.y <= 1 and not pd.isna(lm.x) and not pd.isna(lm.y)

            lmkd = mp_pose.PoseLandmark
            needed = [lmkd.LEFT_SHOULDER, lmkd.LEFT_HIP, lmkd.LEFT_ANKLE, lmkd.LEFT_ELBOW, lmkd.LEFT_WRIST]
            if not all(valid_landmark(landmarks[l.value]) for l in needed):
                shoulder_wrist_vertical_diff = None
            else:
                shoulder_l = [landmarks[lmkd.LEFT_SHOULDER.value].x, landmarks[lmkd.LEFT_SHOULDER.value].y]
                hip_l = [landmarks[lmkd.LEFT_HIP.value].x, landmarks[lmkd.LEFT_HIP.value].y]
                ankle_l = [landmarks[lmkd.LEFT_ANKLE.value].x, landmarks[lmkd.LEFT_ANKLE.value].y]
                elbow_l = [landmarks[lmkd.LEFT_ELBOW.value].x, landmarks[lmkd.LEFT_ELBOW.value].y]
                wrist_l = [landmarks[lmkd.LEFT_WRIST.value].x, landmarks[lmkd.LEFT_WRIST.value].y]

                body_angle = calculate_angle(shoulder_l, hip_l, ankle_l)
                hip_shoulder_vertical_diff = hip_l[1] - shoulder_l[1]
                hip_ankle_vertical_diff = hip_l[1] - ankle_l[1]
                shoulder_elbow_angle = calculate_angle(hip_l, shoulder_l, elbow_l)
                wrist_shoulder_hip_angle = calculate_angle(wrist_l, shoulder_l, hip_l)
                shoulder_wrist_vertical_diff = shoulder_l[1] - wrist_l[1]

                avg_elbow_angle_display = shoulder_elbow_angle
                avg_hip_ankle_display = hip_ankle_vertical_diff
            if 'shoulder_wrist_vertical_diff' in locals() and shoulder_wrist_vertical_diff is not None:
                last_shoulder_wrist_vertical_diff = f"{shoulder_wrist_vertical_diff:.3f}"
                
                signal_buffer.append(shoulder_wrist_vertical_diff)
                features_buffer.append({
                    'body_angle': body_angle,
                    'hip_shoulder_vertical_diff': hip_shoulder_vertical_diff,
                    'hip_ankle_vertical_diff': hip_ankle_vertical_diff,
                    'shoulder_elbow_angle': shoulder_elbow_angle,
                    'wrist_shoulder_hip_angle': wrist_shoulder_hip_angle,
                    'shoulder_wrist_vertical_diff': shoulder_wrist_vertical_diff
                })
                frame_count += 1
                
                if len(signal_buffer) >= 50:
                    signal_array = np.array(signal_buffer)
                    
                    # Suavizar señal con Savitzky-Golay (ventana pequeña para tiempo real)
                    window_length = min(11, len(signal_array) if len(signal_array) % 2 == 1 else len(signal_array) - 1)
                    if window_length >= 5:
                        smoothed_signal = savgol_filter(signal_array, window_length=window_length, polyorder=3)
                    else:
                        smoothed_signal = signal_array
                    
                    signal_range = smoothed_signal.max() - smoothed_signal.min()
                    signal_min = smoothed_signal.min()
                    
                    if signal_range > 0.05:
                        height_threshold = signal_min + (signal_range * 0.40)
                        prominence_min = max(signal_range * 0.10, MIN_PROMINENCE)
                        
                        peaks, properties = find_peaks(
                            smoothed_signal,
                            height=height_threshold,
                            distance=PEAK_MIN_DISTANCE,
                            prominence=prominence_min
                        )
                        
                        for peak_idx in peaks:
                            global_peak_idx = frame_count - len(signal_buffer) + peak_idx
                            
                            if global_peak_idx not in detected_peaks and (global_peak_idx - last_peak_frame) >= PEAK_MIN_DISTANCE:
                                frames_after_peak = len(signal_buffer) - peak_idx - 1
                                
                                if frames_after_peak >= MARGIN_AFTER:
                                    start_idx = max(0, peak_idx - MARGIN_BEFORE)
                                    end_idx = min(len(features_buffer), peak_idx + MARGIN_AFTER)
                                    
                                    window_features = list(features_buffer)[start_idx:end_idx]
                                    
                                    if len(window_features) >= 30:
                                        df_rep = pd.DataFrame(window_features)
                                        
                                        sw_range = df_rep['shoulder_wrist_vertical_diff'].max() - df_rep['shoulder_wrist_vertical_diff'].min()
                                        
                                        if sw_range >= 0.05:
                                            rep_counter += 1
                                            detected_peaks.append(global_peak_idx)
                                            last_peak_frame = global_peak_idx
                                            feedback = f'Rep {rep_counter} completada!'
                                            
                                            feature_columns = [
                                                'body_angle',
                                                'hip_shoulder_vertical_diff',
                                                'hip_ankle_vertical_diff',
                                                'shoulder_elbow_angle',
                                                'wrist_shoulder_hip_angle',
                                                'shoulder_wrist_vertical_diff'
                                            ]
                                            
                                            features_to_predict = {}
                                            for col in feature_columns:
                                                if col in df_rep and not df_rep[col].dropna().empty:
                                                    series = df_rep[col]
                                                    features_to_predict[f'{col}_mean'] = series.mean()
                                                    features_to_predict[f'{col}_std'] = series.std()
                                                    features_to_predict[f'{col}_min'] = series.min()
                                                    features_to_predict[f'{col}_max'] = series.max()
                                                    features_to_predict[f'{col}_range'] = series.max() - series.min()
                                                else:
                                                    features_to_predict[f'{col}_mean'] = 0
                                                    features_to_predict[f'{col}_std'] = 0
                                                    features_to_predict[f'{col}_min'] = 0
                                                    features_to_predict[f'{col}_max'] = 0
                                                    features_to_predict[f'{col}_range'] = 0
                                            
                                            try:
                                                features_scaled = scaler.transform(pd.DataFrame([features_to_predict]))
                                                probabilities = model.predict_proba(features_scaled)[0]
                                                
                                                debug_features = features_to_predict.copy()
                                                debug_probabilities = probabilities.copy()
                                                
                                                adjusted_probabilities = probabilities.copy()
                                                for i, clase in enumerate(le.classes_):
                                                    multiplier = SENSIBILIDAD_CLASE.get(clase, 1.0)
                                                    adjusted_probabilities[i] *= multiplier
                                                
                                                if np.sum(adjusted_probabilities) > 0:
                                                    adjusted_probabilities /= np.sum(adjusted_probabilities)
                                                
                                                last_probabilities = adjusted_probabilities
                                                
                                                prediction_idx = np.argmax(adjusted_probabilities)
                                                prediction = le.classes_[prediction_idx]
                                                confidence = adjusted_probabilities[prediction_idx]
                                                
                                                prediction_clean = prediction.replace('pushup_', '').replace('_', ' ').title()
                                                last_rep_feedback = f"Rep {rep_counter}: {prediction_clean} ({confidence*100:.0f}%)"
                                                
                                            except Exception as e:
                                                probabilities = np.zeros(len(le.classes_))
                                                debug_features = {'error': str(e)}
                                                debug_probabilities = np.zeros(len(le.classes_))
                                                last_rep_feedback = f"Rep {rep_counter}: Error"
                                                print(f"Error en predicción: {str(e)}")
            else:
                last_shoulder_wrist_vertical_diff = 'N/A'

            current_feedback = 'Listo - Haz flexiones'
            if len(signal_buffer) >= 50:
                current_feedback = 'Detectando...'
            
            if last_rep_feedback and last_rep_feedback != 'Ninguna':
                current_feedback = last_rep_feedback

        except Exception as e:
            current_feedback = "Buscando cuerpo..."
            pass

        y_pos = 40
        cv2.putText(dashboard, "REPETICIONES", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y_pos += 45
        cv2.putText(dashboard, str(rep_counter), (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
        y_pos += 60
        cv2.putText(dashboard, "BUFFER", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y_pos += 40
        buffer_info = f"{len(signal_buffer)}/{BUFFER_SIZE}"
        cv2.putText(dashboard, buffer_info, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        y_pos += 60
        display_feedback = current_feedback
        is_error = 'correcto' not in display_feedback.lower() and 'listo' not in display_feedback.lower()
        color_feedback = (74, 69, 255) if is_error else (112, 224, 133)
        cv2.putText(dashboard, "FEEDBACK", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y_pos += 35
        feedback_lines = display_feedback.split(' ')
        if len(feedback_lines) > 2:
            cv2.putText(dashboard, " ".join(feedback_lines[0:2]), (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_feedback, 2)
            if len(feedback_lines) > 2:
                cv2.putText(dashboard, " ".join(feedback_lines[2:]), (15, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_feedback, 2)
            y_pos += 70
        else:
            cv2.putText(dashboard, display_feedback, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_feedback, 2)
            y_pos += 70

        cv2.putText(dashboard, "PREDICCION (Ultima Rep):", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y_pos += 30
        last_prediction_class = ""
        if (last_rep_feedback == 'Ninguna' or np.all(last_probabilities == 0)):
            cv2.putText(dashboard, "Sin prediccion", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            y_pos += 30
            for i, clase in enumerate(le.classes_):
                clase_nombre = str(clase)
                display_name = clase_nombre.replace('pushup_','').replace('_',' ').upper()
                prob_texto = f"{display_name}: 0.0%"
                cv2.putText(dashboard, prob_texto, (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)
                y_pos += 30
        else:
            if last_rep_feedback and ':' in last_rep_feedback:
                last_prediction_class = last_rep_feedback.split(': ')[-1].replace(' ', '_').lower()
            for i, clase in enumerate(le.classes_):
                clase_nombre = str(clase)
                display_name = clase_nombre.replace('pushup_','').replace('_',' ').upper()
                prob_texto = f"{display_name}: {last_probabilities[i] * 100:.1f}%"
                is_last_pred = last_prediction_class in clase_nombre.lower()
                color_pred = (255,255,0) if is_last_pred else (255,255,255)
                grosor = 2 if is_last_pred else 1
                cv2.putText(dashboard, prob_texto, (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_pred, grosor)
                y_pos += 30
        y_pos += 20
        cv2.putText(dashboard, "ANGULOS (DEBUG)", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y_pos += 30
        cv2.putText(dashboard, f"Codo: {int(avg_elbow_angle_display)}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        y_pos += 30
        cv2.putText(dashboard, f"Cadera-Tobillo: {avg_hip_ankle_display:.3f}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        y_pos += 30
        try:
            sw_val = float(last_shoulder_wrist_vertical_diff) if last_shoulder_wrist_vertical_diff != 'N/A' else -999
            sw_color = (0,255,0) if sw_val > -0.20 else (0,0,255)
        except:
            sw_color = (128,128,128)
        cv2.putText(dashboard, f"Hombro-Muneca: {last_shoulder_wrist_vertical_diff}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sw_color, 2)

        y_pos += 40
        cv2.putText(dashboard, "DEBUG FEATURES", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        y_pos += 25
        if debug_features is not None:
            for k, v in debug_features.items():
                cv2.putText(dashboard, f"{k[:12]}: {str(v)[:8]}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                y_pos += 18
        y_pos += 10
        cv2.putText(dashboard, "DEBUG PROBS", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        y_pos += 25
        if debug_probabilities is not None:
            for i, p in enumerate(debug_probabilities):
                cv2.putText(dashboard, f"{i}: {p:.3f}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                y_pos += 18

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )
        output_frame = np.concatenate((image, dashboard), axis=1)
        cv2.imshow('FitTracker AI - Pushup Real-Time Feedback', output_frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()