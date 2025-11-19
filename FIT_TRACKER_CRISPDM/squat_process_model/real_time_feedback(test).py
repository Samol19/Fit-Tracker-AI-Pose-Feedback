import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
import os


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle


# Carga de archivos
try:
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, 'squat_classifier_model.pkl')
    encoder_path = os.path.join(script_dir, 'squat_label_encoder.pkl')
    scaler_path = os.path.join(script_dir, 'squat_scaler.pkl')
    model = joblib.load(model_path)
    le = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    print("Modelo, encoder y scaler cargados correctamente.")
except FileNotFoundError as e:
    print(f"Error: Model, encoder or scaler file not found. Please run train_model.py first.")
    print(f"Details: {e}")
    exit()

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

#configuración de sensibilidad por clase
SENSIBILIDAD_CLASE = {
    'squat_correcto': 1.0,
    'squat_poca_profundidad': 1.0,
    'squat_espalda_arqueada': 1.0,
    'squat_valgo_rodilla': 1.0,
}
for clase_nombre in le.classes_:
    if clase_nombre not in SENSIBILIDAD_CLASE:
        print(f"ADVERTENCIA: La clase '{clase_nombre}' no está en SENSIBILIDAD_CLASE. Usando multiplicador de 1.0")
        SENSIBILIDAD_CLASE[clase_nombre] = 1.0

# Webcam setup
cap = cv2.VideoCapture(0)

DASHBOARD_WIDTH = 400
COLORES_BGR = {
    "blanco": (255, 255, 255),
    "negro": (0, 0, 0),
    "gris_fondo": (41, 41, 41),
    "verde_ok": (112, 224, 133),
    "rojo_error": (74, 69, 255),
    "cian_debug": (255, 255, 0)
}

# State variables
rep_counter = 0
squat_state = 'up'
feedback = 'Listo para empezar'
last_rep_feedback = 'Ninguna' # Para mantener el feedback de la última rep
last_probabilities = np.zeros(len(model.classes_))
current_rep_data = []
avg_knee_angle_display = 0
avg_hip_angle_display = 0

print("Starting real-time feedback. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # dashboard setup
    h_cam, w_cam, _ = frame.shape
    dashboard = np.zeros((h_cam, DASHBOARD_WIDTH, 3), dtype="uint8")
    dashboard[:] = COLORES_BGR["gris_fondo"]

    # Recolor 
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
  
    #detection
    results = pose.process(image)

    # Recolor 
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        landmarks = results.pose_landmarks.landmark
        
        #extraección de keypoints
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        #calculo de ángulos
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        
        avg_knee_angle_display = (left_knee_angle + right_knee_angle) / 2
        avg_hip_angle_display = (left_hip_angle + right_hip_angle) / 2


        # Maquina de estados
        if avg_knee_angle_display < 160 and squat_state == 'up':
            squat_state = 'down'
            feedback = 'Bajando...'
            current_rep_data = [] 
        
        if squat_state == 'down':
            current_rep_data.append({
                'left_knee_angle': left_knee_angle,
                'right_knee_angle': right_knee_angle,
                'left_hip_angle': left_hip_angle,
                'right_hip_angle': right_hip_angle,
                'knee_distance': abs(left_knee[0] - right_knee[0]),
                'hip_shoulder_distance': abs(left_hip[0] - left_shoulder[0])
            })

        if avg_knee_angle_display > 170 and squat_state == 'down':
            squat_state = 'up'
            rep_counter += 1
            feedback = 'Subiendo...'
            
            #Analizar repetición completa
            if current_rep_data:
                rep_df = pd.DataFrame(current_rep_data)
                
                # Calcular promedios de ángulos
                rep_df['avg_knee_angle'] = rep_df[['left_knee_angle', 'right_knee_angle']].mean(axis=1)
                rep_df['avg_hip_angle'] = rep_df[['left_hip_angle', 'right_hip_angle']].mean(axis=1)
                
                # Feature columns a procesar (mismo orden que en build_training_dataset.py)
                feature_columns = ['avg_knee_angle', 'avg_hip_angle', 'knee_distance', 'hip_shoulder_distance']
                
                features_to_predict = {}
                
                # Calcular 5 estadísticas para cada feature
                for col in feature_columns:
                    if col in rep_df.columns and not rep_df[col].dropna().empty:
                        series = rep_df[col]
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
                
                # Preddicción de la repetición
                features_scaled = scaler.transform(pd.DataFrame([features_to_predict]))
                probabilities = model.predict_proba(features_scaled)[0]
                
                adjusted_probabilities = probabilities.copy()
                for i, clase_nombre in enumerate(le.classes_):
                    multiplier = SENSIBILIDAD_CLASE.get(clase_nombre, 1.0)
                    adjusted_probabilities[i] *= multiplier
                
                if np.sum(adjusted_probabilities) > 0:
                    adjusted_probabilities /= np.sum(adjusted_probabilities)
                
                last_probabilities = adjusted_probabilities

                prediction_idx = np.argmax(adjusted_probabilities)
                prediction = le.classes_[prediction_idx]

                last_rep_feedback = f"Rep {rep_counter}: {prediction.replace('squat_', '').replace('_', ' ')}"
        
        # Actualizar feedback en vivo
        if squat_state == 'up':
            current_feedback = 'Listo para bajar'
        elif squat_state == 'down':
            current_feedback = 'Bajando...'
        
        # En la subida, antes de analizar la siguiente repetición, mostrar el último resultado
        if 'Subiendo' in feedback and last_rep_feedback:
             current_feedback = last_rep_feedback
        else:
             current_feedback = feedback if not last_rep_feedback else last_rep_feedback

    except Exception as e:
        current_feedback = "Buscando cuerpo..."
        pass
    
    # display
    y_pos = 40

    # 1. REPS
    cv2.putText(dashboard, "REPETICIONES", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORES_BGR["blanco"], 2)
    y_pos += 45
    cv2.putText(dashboard, str(rep_counter), (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORES_BGR["blanco"], 3)
    y_pos += 60

    # 2. ESTADO
    cv2.putText(dashboard, "ESTADO", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORES_BGR["blanco"], 2)
    y_pos += 40
    cv2.putText(dashboard, squat_state.upper(), (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORES_BGR["blanco"], 2)
    y_pos += 60

    # 3. FEEDBACK
    display_feedback = last_rep_feedback if last_rep_feedback != 'Ninguna' else feedback
    is_error = 'correcto' not in display_feedback and 'listo' not in display_feedback.lower()
    color_feedback = COLORES_BGR["rojo_error"] if is_error else COLORES_BGR["verde_ok"]
    cv2.putText(dashboard, "FEEDBACK", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORES_BGR["blanco"], 2)
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

    # 4. PREDICCION (DEBUG)
    cv2.putText(dashboard, "PREDICCION (Ultima Rep):", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORES_BGR["blanco"], 2)
    y_pos += 30
    
    last_prediction_class = display_feedback.split(': ')[-1].replace(' ', '_') if ':' in display_feedback else ''

    for i, clase_nombre in enumerate(le.classes_):
        display_name = clase_nombre.replace('squat_','').replace('_',' ').upper()
        prob_texto = f"{display_name}: {last_probabilities[i] * 100:.1f}%"

        is_last_pred = last_prediction_class in str(clase_nombre)
        color_pred = COLORES_BGR["cian_debug"] if is_last_pred else COLORES_BGR["blanco"]
        grosor = 2 if is_last_pred else 1

        cv2.putText(dashboard, prob_texto, (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_pred, grosor)
        y_pos += 30

    # 5. ANGULOS (DEBUG)
    y_pos += 20
    cv2.putText(dashboard, "ANGULOS (DEBUG)", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORES_BGR["blanco"], 2)
    y_pos += 30
    cv2.putText(dashboard, f"Rodilla: {int(avg_knee_angle_display)}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORES_BGR["cian_debug"], 2)
    y_pos += 30
    cv2.putText(dashboard, f"Cadera: {int(avg_hip_angle_display)}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORES_BGR["cian_debug"], 2)

    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                             )               

    output_frame = np.concatenate((image, dashboard), axis=1)
    cv2.imshow('FitTracker AI - Real-Time Feedback', output_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()