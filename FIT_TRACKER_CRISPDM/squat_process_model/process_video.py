import cv2
import mediapipe as mp
import numpy as np
import csv
import os

def calculate_angle(a, b, c):
    """Calcula el ángulo entre tres puntos (en grados)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def process_single_video(video_input_path, video_output_path, csv_output_path):
    """Procesa un video para extraer landmarks y ángulos de pose."""
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video en la ruta: {video_input_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

    csv_headers = ['frame']
    for landmark in mp_pose.PoseLandmark:
        name = landmark.name
        csv_headers += [f'{name}_x', f'{name}_y', f'{name}_z', f'{name}_visibility']
    csv_headers += ['left_knee_angle', 'right_knee_angle', 'left_hip_angle', 'right_hip_angle', 'left_shoulder_angle', 'right_shoulder_angle', 'knee_distance', 'hip_shoulder_distance']

    with open(csv_output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

        frame_count = 0
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
                    
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    
                    angle_left_knee = calculate_angle(left_hip, left_knee, left_ankle)
                    angle_right_knee = calculate_angle(right_hip, right_knee, right_ankle)
                    angle_left_hip = calculate_angle(left_shoulder, left_hip, left_knee)
                    angle_right_hip = calculate_angle(right_shoulder, right_hip, right_knee)
                    angle_left_shoulder = calculate_angle(left_elbow, left_shoulder, left_hip)
                    angle_right_shoulder = calculate_angle(right_elbow, right_shoulder, right_hip)
                    
                    knee_distance = abs(left_knee[0] - right_knee[0])
                    hip_shoulder_distance = abs(left_hip[0] - left_shoulder[0])

                    cv2.putText(image, f"L Knee: {int(angle_left_knee)}", tuple(np.multiply(left_knee, [frame_width, frame_height]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"R Knee: {int(angle_right_knee)}", tuple(np.multiply(right_knee, [frame_width, frame_height]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"L Hip: {int(angle_left_hip)}", tuple(np.multiply(left_hip, [frame_width, frame_height]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"R Hip: {int(angle_right_hip)}", tuple(np.multiply(right_hip, [frame_width, frame_height]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    row = [frame_count]
                    lm_row = list(np.array([[l.x, l.y, l.z, l.visibility] for l in landmarks]).flatten())
                    row.extend(lm_row)
                    row.extend([angle_left_knee, angle_right_knee, angle_left_hip, angle_right_hip, angle_left_shoulder, angle_right_shoulder, knee_distance, hip_shoulder_distance])
                    writer.writerow(row)

                except Exception as e:
                    row = [frame_count] + [''] * (len(csv_headers) - 1)
                    writer.writerow(row)
                
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                
                out.write(image)
                frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"-> Procesado: {os.path.basename(video_input_path)}")

