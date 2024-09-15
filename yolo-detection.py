import torch
import numpy as np
from ultralytics import YOLO

# https://github.com/orgs/ultralytics/discussions/8437

camera1_index = ''  # Input your camera
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

labels = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
]

model = YOLO('yolov8n-pose.pt')

for result in model(source=camera1_index, device=device, stream=True, conf=0.3):
    print(result)
