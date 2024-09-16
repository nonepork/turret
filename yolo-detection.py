import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

camera_sources = [2, 4, 6, 8]
caps = [cv2.VideoCapture(src) for src in camera_sources]


def process_frame(frame):
    results = model(frame)
    annotated_frame = results[0].plot()
    return annotated_frame


while True:
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            print(f"無法從攝像頭 {i} 獲取影像")
            continue

        annotated_frame = process_frame(frame)
        cv2.imshow(f"Camera {i}", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
