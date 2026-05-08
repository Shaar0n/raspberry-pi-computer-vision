# TFLite-PiCamera-od-fixed.py
# Edje Electronics + Tejus code, with path fixes and colour conversion for Picamera2 XRGB8888.

import os
import tflite_runtime.interpreter as tflite
import argparse
import cv2
import numpy as np
from picamera2 import Picamera2
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    help='Path to the TFLite file (relative to this script).',
                    default='ssd_mobilenet_v1.tflite')
parser.add_argument('--labels',
                    help='Path to labels file (relative to this script).',
                    default='labels.txt')
parser.add_argument('--threshold',
                    help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution',
                    help='Desired resolution in WxH.',
                    default='1280x720')
args = parser.parse_args()

PATH_TO_MODEL = os.path.join(SCRIPT_DIR, args.model)
PATH_TO_LABELS = os.path.join(SCRIPT_DIR, args.labels)
MIN_CONF_THRESH = float(args.threshold)

resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)

print('Loading model from', PATH_TO_MODEL, '...', end='')
start_time = time.time()

interpreter = tflite.Interpreter(model_path=PATH_TO_MODEL)
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

end_time = time.time()
print('Done! Took {} seconds'.format(end_time - start_time))

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

frame_rate_calc = 1
freq = cv2.getTickFrequency()
print('Running inference for PiCamera')

# ---- Picamera2 init ----
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": 'XRGB8888', "size": (imW, imH)}
    )
)
picam2.start()
time.sleep(1.0)

cv2.namedWindow('Object Detector', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Detector', 1280, 720)

while True:
    current_count = 0
    t1 = cv2.getTickCount()

    frame_raw = picam2.capture_array()
    # Correct conversion from XRGB8888 (BGRA) to BGR
    frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGRA2BGR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if (scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0):
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))
            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame,
                          (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label,
                        (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 2)
            current_count += 1

    cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc),
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 55), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Total Detection Count : ' + str(current_count),
                (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 55), 2, cv2.LINE_AA)

    cv2.imshow('Object Detector', frame)

    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
print("Done")
