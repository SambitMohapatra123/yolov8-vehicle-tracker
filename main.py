import cv2
import pandas as pd
import time
from ultralytics import YOLO
from tracker import *

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Mouse position tracker (optional for debugging)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"x: {x}, y: {y}")
cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

# Load video
cap = cv2.VideoCapture('veh2.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

# Load COCO class names
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Tracker setup
tracker = Tracker()

# Adjusted line positions
line_A_y = 340  # adjusted to be closer to road
line_B_y = 400  # keep as is
real_distance_m = 10  # estimated real-world distance in meters

# Sets and dicts for tracking
going_down_ids = set()
going_up_ids = set()
id_history = {}
entry_time = {}
speed_data = {}

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    data = results[0].boxes.data
    df = pd.DataFrame(data).astype("float")

    detections = []
    for index, row in df.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        class_id = int(row[5])
        class_name = class_list[class_id]
        if class_name in ['car', 'bus']:
            detections.append([x1, y1, x2, y2])

    tracked = tracker.update(detections)

    for x1, y1, x2, y2, obj_id in tracked:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if obj_id in id_history:
            prev_cy = id_history[obj_id]

            # Going down logic
            if prev_cy < line_A_y <= cy and obj_id not in going_down_ids:
                going_down_ids.add(obj_id)
                entry_time[obj_id] = time.time()

            elif prev_cy < line_B_y <= cy and obj_id in entry_time and obj_id not in speed_data:
                time_taken = time.time() - entry_time[obj_id]
                speed = (real_distance_m / time_taken) * 3.6
                speed_data[obj_id] = round(speed, 2)
                print(f"🟢 Going Down: ID {obj_id}, Speed: {speed_data[obj_id]} km/h")

            # Going up logic
            if prev_cy > line_B_y >= cy and obj_id not in going_up_ids:
                going_up_ids.add(obj_id)
                entry_time[obj_id] = time.time()

            elif prev_cy > line_A_y >= cy and obj_id in entry_time and obj_id not in speed_data:
                time_taken = time.time() - entry_time[obj_id]
                speed = (real_distance_m / time_taken) * 3.6
                speed_data[obj_id] = round(speed, 2)
                print(f"🔵 Going Up: ID {obj_id}, Speed: {speed_data[obj_id]} km/h")

        id_history[obj_id] = cy

        # Draw ID and speed
        if obj_id in going_up_ids or obj_id in going_down_ids:
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            text = f"ID:{obj_id}"
            if obj_id in speed_data:
                text += f" | {speed_data[obj_id]} km/h"
            cv2.putText(frame, text, (cx, cy - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2)

    # --- Drawing lines ---
    # Line A (lowered to road level)
    cv2.line(frame, (180, 320), (920, 360), (255, 255, 255), 3)
    cv2.putText(frame, 'Line A', (200, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Line B (unchanged, already good)
    cv2.line(frame, (160, 360), (940, 400), (255, 255, 255), 3)
    cv2.putText(frame, 'Line B', (180, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Count display
    cv2.putText(frame, f'Going Down: {len(going_down_ids)}', (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Going Up: {len(going_up_ids)}', (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show result
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
