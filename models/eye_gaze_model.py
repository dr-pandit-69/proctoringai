from GazeTracking.gaze_tracking import GazeTracking

class EyeGazeModel:
    def __init__(self):
        self.gaze = GazeTracking()

    def estimate_gaze(self, frame):
        self.gaze.refresh(frame)
        frame = self.gaze.annotated_frame()
        text = ""

        if self.gaze.is_blinking():
            text = "Blinking"
        elif self.gaze.is_right():
            text = "Looking right"
        elif self.gaze.is_left():
            text = "Looking left"
        elif self.gaze.is_center():
            text = "Looking center"

        return frame, text
