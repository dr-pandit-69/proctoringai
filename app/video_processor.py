import cv2
from app.ai_processing import process_frame

class VideoProcessor:
    def __init__(self, video_source):
        self.video_source = video_source
        self.vid = cv2.VideoCapture(self.video_source)

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", self.video_source)

        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))

    def process_video(self, output_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        while True:
            ret, frame = self.vid.read()
            if not ret:
                break

            processed_frame = process_frame(frame)
            out.write(processed_frame)

           
            cv2.imshow('Processed Video', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.vid.release()
        out.release()
        cv2.destroyAllWindows()
