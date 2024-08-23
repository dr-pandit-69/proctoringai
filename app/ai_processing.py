import cv2
from joblib import Parallel, delayed
from models.eye_gaze_model import EyeGazeModel
from models.head_detection_model import HeadDetectionModel
from models.emotion_detection_model import EmotionDetectionModel
from models.distance_estimation import DistanceEstimator
from models.audio_processing import AudioProcessor
from models.head_pose_estimation import HeadPoseEstimator
import numpy as np

eye_gaze_model = EyeGazeModel()
head_detection_model = HeadDetectionModel()
emotion_detection_model = EmotionDetectionModel()
distance_estimation_model = DistanceEstimator()
audio_processor = AudioProcessor()
head_pose_estimator = HeadPoseEstimator()

def estimate_gaze(frame):
    return eye_gaze_model.estimate_gaze(frame)

def detect_and_track_faces(frame):
    head_detection_model.label_first_face(frame)
    labeled_face = head_detection_model.track_labeled_face(frame)
    faces = head_detection_model.detect_faces(frame)
    return faces, labeled_face

def detect_emotions(frame, faces):
    emotions = []
    for face in faces:
        x, y, w, h = face['box']
        face_img = frame[y:y+h, x:x+w]
        emotion_label = emotion_detection_model.detect_emotion(face_img)
        emotions.append((x, y, w, h, emotion_label))
    return emotions

def estimate_distance(frame, faces):
    distances = []
    for face in faces:
        x, y, w, h = face['box']
        distance = distance_estimation_model.estimate_distance(w)
        distances.append((x, y, w, h, distance))
    return distances

def process_audio():
    audio_level = audio_processor.get_audio_level()
    noise_detected = audio_level > 25600
    with open('1.txt', 'a') as f:
        f.write(str(audio_level) + '\n')
    return noise_detected

def estimate_head_pose(frame, faces):
    head_poses = []
    for face in faces:
        x, y, w, h = face['box']
        face_img = frame[y:y+h, x:x+w]
        rotation_vector, translation_vector = head_pose_estimator.estimate_head_pose(face_img)
        head_poses.append((x, y, w, h, rotation_vector, translation_vector))
    return head_poses

def process_frame(frame):
    
    gaze_result, head_result, emotion_result, distance_result, head_pose_result, noise_detected = Parallel(n_jobs=6)(
        [delayed(estimate_gaze)(frame), 
         delayed(detect_and_track_faces)(frame),
         delayed(detect_emotions)(frame, head_detection_model.detect_faces(frame)),
         delayed(estimate_distance)(frame, head_detection_model.detect_faces(frame)),
         delayed(estimate_head_pose)(frame, head_detection_model.detect_faces(frame)),
         delayed(process_audio)()
        ]
    )

    frame, gaze_direction = gaze_result
    faces, labeled_face = head_result
    emotions = emotion_result
    distances = distance_result
    head_poses = head_pose_result

    intruder_detected = False

    for face, emotion, distance, head_pose in zip(faces, emotions, distances, head_poses):
        x, y, w, h = face['box']
        ex, ey, ew, eh, emotion_label = emotion
        dx, dy, dw, dh, distance_cm = distance
        hx, hy, hw, hh, rotation_vector, translation_vector = head_pose
        label = 'Intruder'
        color = (0, 0, 255)

        if labeled_face and (face == labeled_face):
            label = 'Student'
            color = (0, 255, 0)
        else:
            intruder_detected = True

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(frame, f'Distance: {distance_cm:.2f} cm', (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

       
        if rotation_vector is not None:
            rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
            proj_matrix = np.hstack((rvec_matrix, translation_vector))
            eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 

            pitch, yaw, roll = [angle.item() for angle in eulerAngles]

            cv2.putText(frame, f'Pitch: {pitch:.2f}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f'Yaw: {yaw:.2f}', (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f'Roll: {roll:.2f}', (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(frame, gaze_direction, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    height = frame.shape[0]
    if noise_detected:
        cv2.putText(frame, "Noise Detected, Please Maintain Silence", (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if intruder_detected:
        cv2.putText(frame, "Intruder Detected, Please be in Isolation", (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if intruder_detected and noise_detected:
        cv2.putText(frame, "Intruder Detected, Please be in Isolation", (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)    

    return frame
