import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

MODEL_URL = "https://tfhub.dev/tensorflow/efficientdet/d0/1"
CONFIDENCE_THRESHOLD = 0.3

print("Loading EfficientDet model... (this may take a moment)")
model = hub.load(MODEL_URL)

def run_efficientdet(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)[tf.newaxis, ...]

    detections = model(input_tensor)

    boxes = detections["detection_boxes"][0].numpy()
    scores = detections["detection_scores"][0].numpy()
    (h, w) = frame.shape[:2]

    confidences = []
    for i in range(len(scores)):
        if scores[i] >= CONFIDENCE_THRESHOLD:
            confidences.append(float(scores[i]))
            y1, x1, y2, x2 = boxes[i]
            x1, x2 = int(x1 * w), int(x2 * w)
            y1, y2 = int(y1 * h), int(y2 * h)
            label = f"{scores[i]:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    precision = float(np.mean(confidences)) if confidences else 0

    return {
        "precision": round(precision, 4),
        "recall": 0,
        "f1": 0,
        "mAP": 0,
        "annotated_frame": frame,
    }