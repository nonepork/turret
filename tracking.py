import cv2
from ultralytics import YOLO

model = YOLO("./yolov8n.onnx")


def detect_person(frame):
    """Run YOLOv8 on a frame and return the bounding box of the first detected person."""
    scale_factor = 0.5  # Reduce frame size to 50%
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

    results = model(small_frame)
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0].item())
            if cls == 0:  # Person detected
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Scale bbox back to original size
                return (int(x1 / scale_factor), int(y1 / scale_factor),
                        int((x2 - x1) / scale_factor), int((y2 - y1) / scale_factor))
    return None


def main():
    cap = cv2.VideoCapture('rtsp://192.168.1.102:8080/h264.sdp')
    tracker = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if tracker is None and frame_count % 10 == 0:  # Run YOLO every 10 frames
            bbox = detect_person(frame)
            if bbox:
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, bbox)
                print("Tracking initialized.")
        else:
            success, bbox = tracker.update(frame) if tracker else (False, None)
            if success:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                print("Tracking lost, re-running YOLO...")
                tracker = None  # Reset tracker to re-detect

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
