# project_final.py
# Real-time face recognition + object detection + CSV + MP4 recording.
# Guarantees ~20 s video by computing FPS from the number of frames.

import os
import cv2
import numpy as np
import face_recognition
import pickle
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter
import time
import csv

# ---------- PATHS ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_ENCODINGS_PATH = os.path.join(SCRIPT_DIR, "encodings.pickle")
TFLITE_MODEL_PATH   = os.path.join(SCRIPT_DIR, "ssd_mobilenet_v1.tflite")
LABELS_PATH         = os.path.join(SCRIPT_DIR, "labels.txt")

CSV_LOG_PATH        = os.path.join(SCRIPT_DIR, "detections_log.csv")
VIDEO_OUTPUT_PATH   = os.path.join(SCRIPT_DIR, "project_run.mp4")
# ---------------------------

RUN_SECONDS   = 20.0          # capture time in seconds
OBJ_THRESHOLD = 0.5
CV_SCALER     = 4             # downscale factor for face_rec
MIN_VIDEO_FPS = 0.5           # safety lower bound

# ---------- LOAD FACE ENCODINGS ----------
print("[INFO] loading face encodings from", FACE_ENCODINGS_PATH)
with open(FACE_ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# ---------- LOAD TFLITE OBJECT MODEL ----------
print("[INFO] loading TFLite model from", TFLITE_MODEL_PATH)
interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_height = input_details[0]['shape'][1]
input_width  = input_details[0]['shape'][2]

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

def detect_objects(frame_bgr):
    """Run TFLite object detection on a BGR frame."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_width, input_height))
    input_data = np.expand_dims(resized, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes  = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    h, w, _ = frame_bgr.shape
    dets = []
    for i in range(len(scores)):
        if scores[i] < OBJ_THRESHOLD:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        left   = int(xmin * w)
        top    = int(ymin * h)
        right  = int(xmax * w)
        bottom = int(ymax * h)
        cls_id = int(classes[i])
        label  = labels[cls_id] if cls_id < len(labels) else str(cls_id)
        conf   = float(scores[i]) * 100.0
        dets.append(
            {"box": (top, right, bottom, left),
             "label": label,
             "conf": conf}
        )
    return dets

# ---------- SETUP CAMERA ----------
print("[INFO] starting camera at 1280x720...")
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (1280, 720)}
    )
)
picam2.start()
time.sleep(2.0)  # AE/AWB settle

# ---------- CSV ----------
csv_file = open(CSV_LOG_PATH, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow([
    "time_s", "type", "label", "confidence",
    "top", "right", "bottom", "left"
])

# List to hold annotated frames for video
video_frames = []

cv2.namedWindow("CS6461 Project - Faces + Objects", cv2.WINDOW_NORMAL)
cv2.resizeWindow("CS6461 Project - Faces + Objects", 1280, 720)

# ---------- MAIN LOOP ----------
print(f"[INFO] capturing for {RUN_SECONDS} seconds...")
run_start = time.time()
while True:
    now = time.time()
    elapsed = now - run_start
    if elapsed >= RUN_SECONDS:
        break

    frame_raw = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame_raw, cv2.COLOR_BGRA2BGR)

    # ----- FACE RECOGNITION -----
    small_bgr = cv2.resize(
        frame_bgr, (0, 0),
        fx=1.0 / CV_SCALER, fy=1.0 / CV_SCALER
    )
    small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(small_rgb)
    face_encodings = face_recognition.face_encodings(
        small_rgb, face_locations, model="small"   # faster than 'large'
    )

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding
        )
        name = "Unknown"
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding
        )
        best_idx = np.argmin(face_distances)
        if matches[best_idx]:
            base_name = known_face_names[best_idx]
            confidence = (1.0 - face_distances[best_idx]) * 100.0
            name_text = f"{base_name} ({confidence:.1f}%)"
            label_only = base_name
        else:
            confidence = 0.0
            name_text = "Unknown"
            label_only = "Unknown"
        face_names.append((name_text, label_only, confidence))

    for (top, right, bottom, left), (name_text, label_only, conf) in zip(
        face_locations, face_names
    ):
        top *= CV_SCALER
        right *= CV_SCALER
        bottom *= CV_SCALER
        left *= CV_SCALER

        cv2.rectangle(frame_bgr, (left, top), (right, bottom),
                      (255, 0, 0), 2)
        cv2.rectangle(frame_bgr, (left - 3, top - 35),
                      (right + 3, top), (255, 0, 0), cv2.FILLED)
        cv2.putText(frame_bgr, name_text,
                    (left + 6, top - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7,
                    (255, 255, 255), 1)

        writer.writerow([
            f"{elapsed:.2f}", "face", label_only,
            f"{conf:.2f}", top, right, bottom, left
        ])

    # ----- OBJECT DETECTION -----
    obj_dets = detect_objects(frame_bgr)
    for det in obj_dets:
        top, right, bottom, left = det["box"]
        label = f'{det["label"]} ({det["conf"]:.1f}%)'
        cv2.rectangle(frame_bgr, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        cv2.putText(frame_bgr, label,
                    (left + 4, top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
        writer.writerow([
            f"{elapsed:.2f}", "object", det["label"],
            f'{det["conf"]:.2f}', top, right, bottom, left
        ])

    # Store annotated frame for later video writing
    video_frames.append(frame_bgr.copy())

    cv2.imshow("CS6461 Project - Faces + Objects", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------- WRITE VIDEO AFTER CAPTURE ----------
frame_count = len(video_frames)
if frame_count > 0:
    video_fps = max(MIN_VIDEO_FPS, frame_count / RUN_SECONDS)  # duration ≈ 20 s [web:43]
    print(f"[INFO] writing {frame_count} frames at {video_fps:.2f} FPS")
    h, w, _ = video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, video_fps, (w, h))
    if not video_writer.isOpened():
        raise RuntimeError("VideoWriter failed to open; codec/container not supported")

    for f in video_frames:
        video_writer.write(f)
    video_writer.release()
else:
    print("[WARN] no frames captured, video not written")

# ---------- CLEANUP ----------
csv_file.close()
picam2.stop()
cv2.destroyAllWindows()
print("[INFO] finished, log saved to", CSV_LOG_PATH)
print("[INFO] MP4 video saved to", VIDEO_OUTPUT_PATH)
