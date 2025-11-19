"""Cliente para feedback en tiempo real con detección local y clasificación remota."""

import warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import numpy as np
import json
import asyncio
import websockets
from collections import deque
import tkinter as tk
from tkinter import ttk
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import threading
import queue

# Configuración local
API_BASE_URL = "ws://34.176.129.163:8080"

ENDPOINTS = {
    "pushup": f"{API_BASE_URL}/ws/pushup",
    "squat": f"{API_BASE_URL}/ws/squat",
    "plank": f"{API_BASE_URL}/ws/plank"
}

PUSHUP_CONFIG = {
    "BUFFER_SIZE": 150,
    "PEAK_MIN_DISTANCE": 25,
    "MARGIN_BEFORE": 20,
    "MARGIN_AFTER": 20,
    "MIN_PROMINENCE": 0.03,
    "MIN_RANGE": 0.05
}

SQUAT_CONFIG = {
    "ANGLE_DOWN": 160,
    "ANGLE_UP": 170
}

PLANK_CONFIG = {
    "BUFFER_SIZE_SECONDS": 1,
    "FPS_ESTIMADO": 30
}

# Funciones de cálculo
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def valid_landmark(lm):
    return lm is not None and 0 <= lm.x <= 1 and 0 <= lm.y <= 1

# Clasificador asíncrono
class AsyncClassifier:
    """Maneja clasificación via WebSocket sin bloquear el video."""
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.is_running = False
        self.thread = None
        
    def start(self):
        """Inicia thread de conexión."""
        self.is_running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
    def _run(self):
        """Loop de conexión WebSocket."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._connect())
        
    async def _connect(self):
        """Mantiene conexión WebSocket."""
        try:
            async with websockets.connect(self.endpoint, ping_timeout=None) as ws:
                print(f"✓ Conectado a {self.endpoint}")
                
                while self.is_running:
                    # Procesar requests
                    try:
                        req_id, frames_list = self.request_queue.get_nowait()
                        
                        print(f"📤 Enviando {len(frames_list)} frames para clasificar")
                        
                        # Enviar array completo de frames al servidor
                        message = {"frames": frames_list}
                        
                        print(f"JSON enviado al API (ID: {req_id}):")
                        print(json.dumps(message, indent=2))
                        
                        await ws.send(json.dumps(message))
                        
                        # Recibir respuesta
                        response = await asyncio.wait_for(ws.recv(), timeout=3.0)
                        data = json.loads(response)
                        
                        print(f"Respuesta: {data.get('prediction', 'error')} ({data.get('confidence', 0)*100:.0f}%)")
                        
                        # Enviar respuesta a la cola
                        self.response_queue.put((req_id, data))
                        
                    except queue.Empty:
                        await asyncio.sleep(0.01)
                    except asyncio.TimeoutError:
                        print(f"Timeout")
                        self.response_queue.put((req_id, {"error": "timeout"}))
                    except Exception as e:
                        print(f"Error: {e}")
                        self.response_queue.put((req_id, {"error": str(e)}))
                        
        except Exception as e:
            print(f"Error de conexión: {e}")
            self.is_running = False
            
    def classify(self, req_id, frames_list):
        """Envía request de clasificación con lista de frames."""
        self.request_queue.put((req_id, frames_list))
        
    def get_response(self):
        """Obtiene respuesta si hay disponible."""
        try:
            return self.response_queue.get_nowait()
        except queue.Empty:
            return None
            
    def stop(self):
        """Detiene el clasificador."""
        self.is_running = False

# Extracción de features
def extract_pushup_features(landmarks, mp_pose):
    """Extrae las 6 features base para pushup."""
    lmkd = mp_pose.PoseLandmark
    
    needed = [lmkd.LEFT_SHOULDER, lmkd.LEFT_HIP, lmkd.LEFT_ANKLE, lmkd.LEFT_ELBOW, lmkd.LEFT_WRIST]
    if not all(valid_landmark(landmarks[l.value]) for l in needed):
        return None
    
    shoulder_l = [landmarks[lmkd.LEFT_SHOULDER.value].x, landmarks[lmkd.LEFT_SHOULDER.value].y]
    hip_l = [landmarks[lmkd.LEFT_HIP.value].x, landmarks[lmkd.LEFT_HIP.value].y]
    ankle_l = [landmarks[lmkd.LEFT_ANKLE.value].x, landmarks[lmkd.LEFT_ANKLE.value].y]
    elbow_l = [landmarks[lmkd.LEFT_ELBOW.value].x, landmarks[lmkd.LEFT_ELBOW.value].y]
    wrist_l = [landmarks[lmkd.LEFT_WRIST.value].x, landmarks[lmkd.LEFT_WRIST.value].y]
    
    return {
        "body_angle": calculate_angle(shoulder_l, hip_l, ankle_l),
        "hip_shoulder_vertical_diff": hip_l[1] - shoulder_l[1],
        "hip_ankle_vertical_diff": hip_l[1] - ankle_l[1],
        "shoulder_elbow_angle": calculate_angle(hip_l, shoulder_l, elbow_l),
        "wrist_shoulder_hip_angle": calculate_angle(wrist_l, shoulder_l, hip_l),
        "shoulder_wrist_vertical_diff": shoulder_l[1] - wrist_l[1]
    }

def extract_squat_features(landmarks, mp_pose):
    """Extrae los ángulos y features para squat."""
    lmkd = mp_pose.PoseLandmark
    
    needed = [lmkd.LEFT_SHOULDER, lmkd.RIGHT_SHOULDER, lmkd.LEFT_HIP, lmkd.RIGHT_HIP,
              lmkd.LEFT_KNEE, lmkd.RIGHT_KNEE, lmkd.LEFT_ANKLE, lmkd.RIGHT_ANKLE]
    if not all(valid_landmark(landmarks[l.value]) for l in needed):
        return None
    
    shoulder_l = [landmarks[lmkd.LEFT_SHOULDER.value].x, landmarks[lmkd.LEFT_SHOULDER.value].y]
    shoulder_r = [landmarks[lmkd.RIGHT_SHOULDER.value].x, landmarks[lmkd.RIGHT_SHOULDER.value].y]
    hip_l = [landmarks[lmkd.LEFT_HIP.value].x, landmarks[lmkd.LEFT_HIP.value].y]
    hip_r = [landmarks[lmkd.RIGHT_HIP.value].x, landmarks[lmkd.RIGHT_HIP.value].y]
    knee_l = [landmarks[lmkd.LEFT_KNEE.value].x, landmarks[lmkd.LEFT_KNEE.value].y]
    knee_r = [landmarks[lmkd.RIGHT_KNEE.value].x, landmarks[lmkd.RIGHT_KNEE.value].y]
    ankle_l = [landmarks[lmkd.LEFT_ANKLE.value].x, landmarks[lmkd.LEFT_ANKLE.value].y]
    ankle_r = [landmarks[lmkd.RIGHT_ANKLE.value].x, landmarks[lmkd.RIGHT_ANKLE.value].y]
    
    left_knee_angle = calculate_angle(hip_l, knee_l, ankle_l)
    right_knee_angle = calculate_angle(hip_r, knee_r, ankle_r)
    left_hip_angle = calculate_angle(shoulder_l, hip_l, knee_l)
    right_hip_angle = calculate_angle(shoulder_r, hip_r, knee_r)
    
    return {
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "left_hip_angle": left_hip_angle,
        "right_hip_angle": right_hip_angle,
        "knee_distance": abs(knee_l[0] - knee_r[0]),
        "hip_shoulder_distance": abs(hip_l[0] - shoulder_l[0]),
        "avg_knee_angle": (left_knee_angle + right_knee_angle) / 2,
        "avg_hip_angle": (left_hip_angle + right_hip_angle) / 2
    }

def extract_plank_features(landmarks, mp_pose):
    """Extrae las 5 features base para plank."""
    lmkd = mp_pose.PoseLandmark
    
    needed = [lmkd.LEFT_SHOULDER, lmkd.LEFT_ELBOW, lmkd.LEFT_HIP, lmkd.LEFT_ANKLE, lmkd.LEFT_WRIST]
    if not all(valid_landmark(landmarks[l.value]) for l in needed):
        return None
    
    shoulder_l = [landmarks[lmkd.LEFT_SHOULDER.value].x, landmarks[lmkd.LEFT_SHOULDER.value].y]
    elbow_l = [landmarks[lmkd.LEFT_ELBOW.value].x, landmarks[lmkd.LEFT_ELBOW.value].y]
    hip_l = [landmarks[lmkd.LEFT_HIP.value].x, landmarks[lmkd.LEFT_HIP.value].y]
    ankle_l = [landmarks[lmkd.LEFT_ANKLE.value].x, landmarks[lmkd.LEFT_ANKLE.value].y]
    wrist_l = [landmarks[lmkd.LEFT_WRIST.value].x, landmarks[lmkd.LEFT_WRIST.value].y]
    
    return {
        "body_angle": calculate_angle(shoulder_l, hip_l, ankle_l),
        "hip_shoulder_vertical_diff": hip_l[1] - shoulder_l[1],
        "hip_ankle_vertical_diff": hip_l[1] - ankle_l[1],
        "shoulder_elbow_angle": calculate_angle(hip_l, shoulder_l, elbow_l),
        "wrist_shoulder_hip_angle": calculate_angle(wrist_l, shoulder_l, hip_l)
    }

# Clase principal
class FitTrackerApp:
    def __init__(self):
        self.current_exercise = "pushup"
        self.classifier = None
        self.is_running = False
        self.rep_counter = 0
        self.last_feedback = "Selecciona un ejercicio"
        self.last_probabilities = {}
        self.pending_req_id = None
        
        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Pushup: detección por picos
        self.pushup_signal_buffer = deque(maxlen=PUSHUP_CONFIG["BUFFER_SIZE"])
        self.pushup_features_buffer = deque(maxlen=PUSHUP_CONFIG["BUFFER_SIZE"])
        self.pushup_detected_peaks = []
        self.pushup_last_peak_frame = -50
        self.pushup_frame_count = 0
        
        # Squat: state machine
        self.squat_state = 'up'
        self.squat_current_rep_data = []
        
        # Plank: buffer temporal
        self.plank_feature_buffer = []
        
        # Frame tracking
        self.frame_count = 0
        self.last_feedback_frame = 0
        
        # Debug
        self.avg_knee_angle_display = 0
        self.avg_hip_angle_display = 0
        self.last_shoulder_wrist_vertical_diff = 'N/A'
        
        self.create_control_window()
        
    def create_control_window(self):
        """Crea ventana de control."""
        self.root = tk.Tk()
        self.root.title("FitTracker AI - Control")
        self.root.geometry("400x250")
        self.root.resizable(False, False)
        
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        title = ttk.Label(main_frame, text="Selecciona el Ejercicio", font=("Arial", 16, "bold"))
        title.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        btn_pushup = ttk.Button(main_frame, text="🏋️ PUSHUP", 
                               command=lambda: self.change_exercise("pushup"), width=12)
        btn_pushup.grid(row=1, column=0, padx=5, pady=10)
        
        btn_squat = ttk.Button(main_frame, text="🦵 SQUAT", 
                              command=lambda: self.change_exercise("squat"), width=12)
        btn_squat.grid(row=1, column=1, padx=5, pady=10)
        
        btn_plank = ttk.Button(main_frame, text="🧘 PLANK", 
                              command=lambda: self.change_exercise("plank"), width=12)
        btn_plank.grid(row=1, column=2, padx=5, pady=10)
        
        self.exercise_label = ttk.Label(main_frame, text=f"Actual: {self.current_exercise.upper()}", 
                                       font=("Arial", 12))
        self.exercise_label.grid(row=2, column=0, columnspan=3, pady=20)
        
        self.start_btn = ttk.Button(main_frame, text="▶ INICIAR", 
                                    command=self.toggle_camera, width=30)
        self.start_btn.grid(row=3, column=0, columnspan=3, pady=10)
        
        info = ttk.Label(main_frame, text="Presiona 'q' en la ventana de video para salir",
                        font=("Arial", 9), foreground="gray")
        info.grid(row=4, column=0, columnspan=3, pady=(10, 0))
        
    def change_exercise(self, exercise):
        """Cambia el ejercicio actual."""
        self.current_exercise = exercise
        self.exercise_label.config(text=f"Actual: {exercise.upper()}")
        self.rep_counter = 0
        self.last_feedback = f"Preparado para {exercise}"
        self.last_probabilities = {}
        self.pending_req_id = None
        
        # Resetear buffers
        if exercise == "pushup":
            self.pushup_signal_buffer.clear()
            self.pushup_features_buffer.clear()
            self.pushup_detected_peaks.clear()
            self.pushup_last_peak_frame = -50
            self.pushup_frame_count = 0
        elif exercise == "squat":
            self.squat_state = 'up'
            self.squat_current_rep_data.clear()
        elif exercise == "plank":
            self.plank_feature_buffer.clear()
        
        # Reconectar clasificador
        if self.is_running and self.classifier:
            self.classifier.stop()
            self.classifier = AsyncClassifier(ENDPOINTS[exercise])
            self.classifier.start()
        
        print(f"✓ Ejercicio cambiado a: {exercise.upper()}")
        
    def toggle_camera(self):
        """Inicia/detiene la cámara."""
        if not self.is_running:
            self.is_running = True
            self.start_btn.config(text="⏸ DETENER")
            
            # Iniciar clasificador
            self.classifier = AsyncClassifier(ENDPOINTS[self.current_exercise])
            self.classifier.start()
            
            # Ejecutar cámara en thread
            camera_thread = threading.Thread(target=self.run_camera, daemon=True)
            camera_thread.start()
        else:
            self.is_running = False
            self.start_btn.config(text="▶ INICIAR")
            if self.classifier:
                self.classifier.stop()
                
    def run_camera(self):
        """Loop principal de cámara."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        try:
            while cap.isOpened() and self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.process_frame(frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
                    break
                        
        except Exception as e:
            print(f"Error: {e}")
            self.last_feedback = f"Error: {str(e)}"
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.is_running = False
            self.start_btn.config(text="▶ INICIAR")
            
    def process_frame(self, frame):
        """Procesa cada frame con lógica específica por ejercicio."""
        self.frame_count += 1
        
        h_cam, w_cam, _ = frame.shape
        dashboard_width = 450
        dashboard = np.zeros((h_cam, dashboard_width, 3), dtype="uint8")
        dashboard[:] = (41, 41, 41)
        
        # Procesar con MediaPipe (optimizado)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        current_feedback = "Buscando cuerpo..."
        
        # ==================== LÓGICA POR EJERCICIO ====================
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            if self.current_exercise == "pushup":
                current_feedback = self.process_pushup(landmarks)
            elif self.current_exercise == "squat":
                current_feedback = self.process_squat(landmarks)
            elif self.current_exercise == "plank":
                current_feedback = self.process_plank(landmarks)
        
        # Verificar respuestas del clasificador
        response = self.classifier.get_response()
        if response:
            req_id, data = response
            
            if req_id == self.pending_req_id:
                if "error" not in data and "prediction" in data:
                    prediction = data["prediction"].replace(f'{self.current_exercise}_', '').replace('_', ' ').title()
                    confidence = data.get("confidence", 0)
                    
                    if self.current_exercise == "plank":
                        self.last_feedback = f"{prediction} ({confidence*100:.0f}%)"
                    else:
                        self.last_feedback = f"Rep {self.rep_counter}: {prediction} ({confidence*100:.0f}%)"
                    
                    self.last_probabilities = data.get("probabilities", {})
                    self.last_feedback_frame = self.frame_count
                else:
                    print(f"Error: {data.get('error', 'unknown')}")
                    
                self.pending_req_id = None
        
        # Mantener última clasificación visible
        if self.last_feedback and self.last_feedback != "Selecciona un ejercicio":
            current_feedback = self.last_feedback
        
        # ==================== DIBUJAR DASHBOARD ====================
        self.draw_dashboard(dashboard, current_feedback, h_cam)
        
        # Dibujar esqueleto
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        
        output_frame = np.concatenate((image, dashboard), axis=1)
        cv2.imshow('FitTracker AI - Real-Time Feedback', output_frame)
        
    def process_pushup(self, landmarks):
        """Lógica de detección para PUSHUP (igual que real_time_feedback.py original)."""
        features = extract_pushup_features(landmarks, self.mp_pose)
        if not features:
            return "Posición no detectada"
        
        shoulder_wrist_vertical_diff = features["shoulder_wrist_vertical_diff"]
        self.last_shoulder_wrist_vertical_diff = f"{shoulder_wrist_vertical_diff:.3f}"
        
        # Agregar a buffers
        self.pushup_signal_buffer.append(shoulder_wrist_vertical_diff)
        self.pushup_features_buffer.append(features)
        self.pushup_frame_count += 1
        
        # Detección de picos
        if len(self.pushup_signal_buffer) >= 50:
            signal_array = np.array(self.pushup_signal_buffer)
            
            window_length = min(11, len(signal_array) if len(signal_array) % 2 == 1 else len(signal_array) - 1)
            if window_length >= 5:
                smoothed_signal = savgol_filter(signal_array, window_length=window_length, polyorder=3)
            else:
                smoothed_signal = signal_array
            
            signal_range = smoothed_signal.max() - smoothed_signal.min()
            signal_min = smoothed_signal.min()
            
            if signal_range > 0.05:
                height_threshold = signal_min + (signal_range * 0.40)
                prominence_min = max(signal_range * 0.10, PUSHUP_CONFIG["MIN_PROMINENCE"])
                
                peaks, properties = find_peaks(
                    smoothed_signal,
                    height=height_threshold,
                    distance=PUSHUP_CONFIG["PEAK_MIN_DISTANCE"],
                    prominence=prominence_min
                )
                
                for peak_idx in peaks:
                    global_peak_idx = self.pushup_frame_count - len(self.pushup_signal_buffer) + peak_idx
                    
                    if global_peak_idx not in self.pushup_detected_peaks and \
                       (global_peak_idx - self.pushup_last_peak_frame) >= PUSHUP_CONFIG["PEAK_MIN_DISTANCE"]:
                        
                        frames_after_peak = len(self.pushup_signal_buffer) - peak_idx - 1
                        
                        if frames_after_peak >= PUSHUP_CONFIG["MARGIN_AFTER"]:
                            start_idx = max(0, peak_idx - PUSHUP_CONFIG["MARGIN_BEFORE"])
                            end_idx = min(len(self.pushup_features_buffer), peak_idx + PUSHUP_CONFIG["MARGIN_AFTER"])
                            
                            window_features = list(self.pushup_features_buffer)[start_idx:end_idx]
                            
                            if len(window_features) >= 30:
                                df_rep = pd.DataFrame(window_features)
                                sw_range = df_rep['shoulder_wrist_vertical_diff'].max() - df_rep['shoulder_wrist_vertical_diff'].min()
                                
                                if sw_range >= PUSHUP_CONFIG["MIN_RANGE"]:
                                    self.rep_counter += 1
                                    self.pushup_detected_peaks.append(global_peak_idx)
                                    self.pushup_last_peak_frame = global_peak_idx
                                    
                                    # Enviar frames completos al servidor (él calcula estadísticas)
                                    frames_list = window_features
                                    
                                    self.pending_req_id = self.rep_counter
                                    self.classifier.classify(self.pending_req_id, frames_list)
                                    
                                    print(f"🔍 PUSHUP Rep {self.rep_counter} detectada, enviando {len(frames_list)} frames")
                                    
                                    return f'Rep {self.rep_counter} detectada! Clasificando...'
        
        return 'Listo - Haz flexiones'
        
    def process_squat(self, landmarks):
        """Lógica de detección para SQUAT (state machine)."""
        features = extract_squat_features(landmarks, self.mp_pose)
        if not features:
            return "Posición no detectada"
        
        avg_knee_angle = features["avg_knee_angle"]
        self.avg_knee_angle_display = avg_knee_angle
        self.avg_hip_angle_display = features["avg_hip_angle"]
        
        # State machine
        if avg_knee_angle < SQUAT_CONFIG["ANGLE_DOWN"] and self.squat_state == 'up':
            self.squat_state = 'down'
            self.squat_current_rep_data = []
        
        if self.squat_state == 'down':
            self.squat_current_rep_data.append({
                'left_knee_angle': features['left_knee_angle'],
                'right_knee_angle': features['right_knee_angle'],
                'left_hip_angle': features['left_hip_angle'],
                'right_hip_angle': features['right_hip_angle'],
                'knee_distance': features['knee_distance'],
                'hip_shoulder_distance': features['hip_shoulder_distance']
            })
        
        if avg_knee_angle > SQUAT_CONFIG["ANGLE_UP"] and self.squat_state == 'down':
            self.squat_state = 'up'
            self.rep_counter += 1
            
            if self.squat_current_rep_data:
                # Calcular avg_knee_angle y avg_hip_angle para cada frame
                frames_list = []
                for frame in self.squat_current_rep_data:
                    frame_copy = frame.copy()
                    frame_copy['avg_knee_angle'] = (frame['left_knee_angle'] + frame['right_knee_angle']) / 2
                    frame_copy['avg_hip_angle'] = (frame['left_hip_angle'] + frame['right_hip_angle']) / 2
                    frames_list.append(frame_copy)
                
                self.pending_req_id = self.rep_counter
                self.classifier.classify(self.pending_req_id, frames_list)
                
                print(f"🔍 SQUAT Rep {self.rep_counter} completada, enviando {len(frames_list)} frames")
                
                return f'Rep {self.rep_counter} completada! Clasificando...'
        
        return 'Listo para bajar' if self.squat_state == 'up' else 'Bajando...'
        
    def process_plank(self, landmarks):
        """Lógica de detección para PLANK (buffer temporal)."""
        features = extract_plank_features(landmarks, self.mp_pose)
        if not features:
            return "Posición no detectada"
        
        self.plank_feature_buffer.append(list(features.values()))
        
        buffer_size = PLANK_CONFIG["BUFFER_SIZE_SECONDS"] * PLANK_CONFIG["FPS_ESTIMADO"]
        
        if len(self.plank_feature_buffer) >= buffer_size:
            # Convertir lista de listas a lista de diccionarios
            frames_list = []
            feature_names = list(features.keys())
            for frame_values in self.plank_feature_buffer:
                frame_dict = dict(zip(feature_names, frame_values))
                frames_list.append(frame_dict)
            
            self.pending_req_id = len(self.plank_feature_buffer)
            self.classifier.classify(self.pending_req_id, frames_list)
            
            print(f"🔍 PLANK Buffer completo, enviando {len(frames_list)} frames")
            
            self.plank_feature_buffer.clear()
            return "Clasificando postura..."
        
        return f"Analizando... ({len(self.plank_feature_buffer)}/{buffer_size})"
        
    def draw_dashboard(self, dashboard, current_feedback, h_cam):
        """Dibuja el dashboard con información."""
        y_pos = 40
        
        # Ejercicio actual
        cv2.putText(dashboard, "EJERCICIO", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y_pos += 45
        cv2.putText(dashboard, self.current_exercise.upper(), (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,0), 3)
        y_pos += 60
        
        # Repeticiones (solo pushup y squat)
        if self.current_exercise in ["pushup", "squat"]:
            cv2.putText(dashboard, "REPETICIONES", (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            y_pos += 45
            cv2.putText(dashboard, str(self.rep_counter), (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
            y_pos += 60
        
        # Feedback
        cv2.putText(dashboard, "FEEDBACK", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y_pos += 35
        
        is_correct = 'correcto' in current_feedback.lower()
        color_feedback = (112, 224, 133) if is_correct else (74, 69, 255)
        
        # Dividir texto largo
        max_chars = 25
        words = current_feedback.split(' ')
        current_line = ""
        for word in words:
            if len(current_line + word) <= max_chars:
                current_line += word + " "
            else:
                cv2.putText(dashboard, current_line.strip(), (15, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_feedback, 2)
                y_pos += 30
                current_line = word + " "
        if current_line:
            cv2.putText(dashboard, current_line.strip(), (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_feedback, 2)
        y_pos += 60
        
        # Probabilidades - SIEMPRE MOSTRAR (aunque sea vacío)
        cv2.putText(dashboard, "PROBABILIDADES:", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y_pos += 35
        
        if self.last_probabilities:
            sorted_probs = sorted(self.last_probabilities.items(), 
                                 key=lambda x: x[1], reverse=True)
            
            for clase, prob in sorted_probs:
                display_name = clase.replace(f'{self.current_exercise}_', '')
                display_name = display_name.replace('_', ' ').title()
                
                prob_texto = f"{display_name}: {prob*100:.1f}%"
                
                is_max = prob == sorted_probs[0][1]
                color = (255, 255, 0) if is_max else (200, 200, 200)
                thickness = 2 if is_max else 1
                
                cv2.putText(dashboard, prob_texto, (25, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, thickness)
                y_pos += 30
        else:
            # Mostrar mensaje si no hay probabilidades aún
            cv2.putText(dashboard, "Sin clasificacion aun", (25, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 1)
            y_pos += 30
        
        # Debug info
        if self.current_exercise == "squat":
            y_pos = h_cam - 100
            cv2.putText(dashboard, "ANGULOS (DEBUG):", (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y_pos += 25
            cv2.putText(dashboard, f"Rodilla: {int(self.avg_knee_angle_display)}", (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            y_pos += 25
            cv2.putText(dashboard, f"Cadera: {int(self.avg_hip_angle_display)}", (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        elif self.current_exercise == "pushup":
            y_pos = h_cam - 80
            cv2.putText(dashboard, f"Buffer: {len(self.pushup_signal_buffer)}/{PUSHUP_CONFIG['BUFFER_SIZE']}", 
                       (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
            y_pos += 25
            cv2.putText(dashboard, f"Señal: {self.last_shoulder_wrist_vertical_diff}", 
                       (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
        
    def run(self):
        """Inicia la aplicación."""
        print("=" * 60)
        print("🏋️ FitTracker AI - Cliente WebSocket")
        print("=" * 60)
        print(f"API: {API_BASE_URL}")
        print("Ejercicios: PUSHUP, SQUAT, PLANK")
        print("=" * 60)
        
        self.root.mainloop()

# Ejecutar la aplicación
if __name__ == '__main__':
    app = FitTrackerApp()
    app.run()
