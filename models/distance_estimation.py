import cv2
import numpy as np
from mtcnn import MTCNN

class DistanceEstimator:
    def __init__(self, focal_length=640, real_face_width=15):  
        self.detector = MTCNN()
        self.focal_length = focal_length
        self.real_face_width = real_face_width

    def detect_faces(self, frame):
        
        faces = self.detector.detect_faces(frame)
        return faces

    def estimate_distance(self, face_width_in_pixels):
        
        distance = (self.real_face_width * self.focal_length) / face_width_in_pixels
        return distance

    def process_frame(self, frame):
        faces = self.detect_faces(frame)
        distances = []
        for face in faces:
            x, y, w, h = face['box']
            distance = self.estimate_distance(w)
            distances.append((x, y, w, h, distance))
        return frame, distances
