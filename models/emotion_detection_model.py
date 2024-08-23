import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

class EmotionDetectionModel:
    def __init__(self):
      
        self.model = load_model('models/emotion_model.hdf5', compile=False)
   
        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def detect_emotion(self, face):
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (64, 64))  # Resize to the correct dimensions
        gray_face = gray_face / 255.0
        gray_face = np.expand_dims(gray_face, axis=0)
        gray_face = np.expand_dims(gray_face, axis=-1)
        
        prediction = self.model.predict(gray_face)
        emotion_label = self.emotion_labels[np.argmax(prediction)]
        return emotion_label
