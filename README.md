
# AI Proctoring System

The AI Proctoring System is an advanced solution designed to enhance the integrity of online examinations by multiple components, including face detection, emotion detection, eye gaze tracking, distance estimation, head pose estimation, and noise detection. This system ensures a secure and monitored exam environment, effectively identifying any suspicious activities such as intruder detection or noise disturbances.

![Demo Sample](https://media.giphy.com/media/iEMoiFXkTGuu9O2Jmz/giphy.gif)



## Features

**Face Detection**: Detects and tracks faces using the MTCNN model, ensuring that only authorized individuals are present during the exam.

**Multiple Person Detection**: Simultaneously detects and tracks multiple faces, ensuring that no unauthorized individuals are present.

**Emotion Detection**: Identifies the examinee's emotional state using a convolutional neural network trained on the FER2013 dataset.

**Eye Gaze Tracking**: Monitors the direction of the examinee's gaze to detect if they are looking away from the screen, indicating potential cheating.

**Distance Estimation**: Estimates the distance of the examinee from the camera to ensure they remain within an acceptable range.
Head Pose Estimation: Determines the pitch, yaw, and roll of the examinee's head, detecting unnatural head movements that could indicate cheating.

**Noise Detection**: Detects ambient noise levels, triggering alerts if excessive noise is detected, which could indicate unauthorized collaboration.

**Intruder Detection**: Identifies and flags any unauthorized individuals who enter the frame during the examination.

## Installation

### Prerequisites

- Python 3.9+
- Virtual Environment such a pipenv (Recommended)
- Required Python Libraries:
  - TensorFlow
  - OpenCV
  - Joblib
  - Keras
  - Numpy
  - Pyaudio (for noise detection)






### Steps to run the Project

1. Clone the repository

```bash
  git clone https://github.com/dr-pandit-69/proctoringai
  cd proctoringai
```

2. Clone the GazeTracking repository (essential)

```bash
git clone https://github.com/antoinelame/GazeTracking
```
    
3. Create a virtual environment (pipenv is used here)

```bash
pipenv shell
```
4. Install the dependencies

```bash
pip install -r requirements.txt
```

5. Run the Project :D

```bash
python3 main.py
```

