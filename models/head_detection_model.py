import cv2
from mtcnn import MTCNN

class HeadDetectionModel:
    def __init__(self):
        self.detector = MTCNN()
        self.labeled_face = None

    def detect_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb_frame)
        return faces

    def label_first_face(self, frame):
        faces = self.detect_faces(frame)
        if faces and self.labeled_face is None:
            self.labeled_face = faces[0]  # Label the first detected face

    def track_labeled_face(self, frame):
        faces = self.detect_faces(frame)
        if not faces:
            return None

        if self.labeled_face:
            lx, ly, lw, lh = self.labeled_face['box']
            for face in faces:
                x, y, w, h = face['box']
                if abs(x - lx) < 50 and abs(y - ly) < 50:  # Check proximity to the labeled face
                    self.labeled_face = face
                    return face
        return None
