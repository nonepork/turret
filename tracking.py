import cv2
from ultralytics import YOLO

model = YOLO("./yolo11n-pose.pt")


def detect_person(frame):
    """Run YOLOv8 on a frame and return the bounding box of the head of the first detected person."""
    scale_factor = 0.5  # Reduce frame size to 50%
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    results = model(small_frame)

    for result in results:
        if result.keypoints is None:
            continue

        # Get keypoints for the first person detected
        # Get keypoints of first person
        keypoints = result.keypoints[0].data[0]

        # YOLO pose keypoints for head are:
        # 0: nose, 1: left eye, 2: right eye, 3: left ear, 4: right ear
        head_points = keypoints[:5]  # Get first 5 keypoints (head points)

        # Filter out low confidence points
        # Only use points with confidence > 0.5
        valid_points = head_points[head_points[:, 2] > 0.5]

        if len(valid_points) < 2:  # Need at least 2 points to create a bounding box
            continue

        # Calculate head bounding box
        x_coords = valid_points[:, 0]
        y_coords = valid_points[:, 1]

        # Create a slightly larger box around the head
        padding = 5  # Add padding around the head
        x1 = int(min(x_coords) - padding)
        y1 = int(min(y_coords) - padding)
        x2 = int(max(x_coords) + padding)
        y2 = int(max(y_coords) + padding)

        # Convert to original scale and return in (x, y, w, h) format
        x = int(x1 / scale_factor)
        y = int(y1 / scale_factor)
        w = int((x2 - x1) / scale_factor)
        h = int((y2 - y1) / scale_factor)

        return (x, y, w, h)

    return None


def main():
    cap = cv2.VideoCapture(0)
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
