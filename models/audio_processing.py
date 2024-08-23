import pyaudio
import numpy as np

class AudioProcessor:
    def __init__(self, rate=16000, chunk=1024, threshold=26000):
        self.rate = rate
        self.chunk = chunk
        self.threshold = threshold
        self.audio_interface = pyaudio.PyAudio()
        self.stream = self.audio_interface.open(format=pyaudio.paInt16,
                                                channels=1,
                                                rate=self.rate,
                                                input=True,
                                                frames_per_buffer=self.chunk)
    
    def get_audio_level(self):
        data = self.stream.read(self.chunk, exception_on_overflow=False)
        signal = np.frombuffer(data, dtype=np.int16)
        level = np.abs(signal).mean()
        return level

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio_interface.terminate()
