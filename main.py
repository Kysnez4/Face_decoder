import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from video_capture import VideoCapture

if __name__ == "__main__":
    video_capture = VideoCapture(detection_interval=50)
    video_capture.start()
