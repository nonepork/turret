import torch
from ultralytics import YOLO

camera1_index = '' # Input your camera
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = YOLO('yolov8n-pose.pt')

for result in model(source=camera1_index, device=device, stream=True, show=True, conf=0.3):
    print(result)
