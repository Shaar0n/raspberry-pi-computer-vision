import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the camera with near-default ISP controls
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": 'XRGB8888', "size": (1920, 1080)}
    )
)

# Use more natural values (or comment out entirely to use full auto)
picam2.set_controls({
    "Brightness": 0.0,   # default-like brightness
    "Contrast": 1.0,     # default-like contrast
    # "Saturation": 1.0, # uncomment if you want to tweak saturation
})

picam2.start()
time.sleep(2.0)  # allow auto exposure/white-balance to settle

cv_scaler = 4  # has to be a whole number

face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

def process_frame(frame):
    global face_locations, face_encodings, face_names

    # Resize for performance
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Face detection and encoding
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_resized_frame, face_locations, model='large'
    )

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding
        )
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            confidence = (1 - face_distances[best_match_index]) * 100
            name = f"{name} ({confidence:.1f})"
        face_names.append(name)

    return frame

def draw_results(frame):
    # Show detection results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up locations
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        # Draw blue box and labels
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.rectangle(
            frame, (left - 3, top - 35), (right + 3, top),
            (255, 0, 0), cv2.FILLED
        )
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(
            frame, name, (left + 6, top - 6),
            font, 0.8, (255, 255, 255), 1
        )

    return frame

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

# Create a resizable and appropriately sized display window
cv2.namedWindow("Face Rec Running", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Rec Running", 1280, 720)

while True:
    frame = picam2.capture_array()
    processed_frame = process_frame(frame)
    display_frame = draw_results(processed_frame)
    current_fps = calculate_fps()

    # Attach FPS counter (green text)
    cv2.putText(
        display_frame, f"FPS: {current_fps:.1f}",
        (display_frame.shape[1] - 150, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    cv2.imshow("Face Rec Running", display_frame)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
picam2.stop()
