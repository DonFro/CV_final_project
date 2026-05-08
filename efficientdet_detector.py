import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import os
from utils.metrics import compute_metrics

# EfficientDet Model downloads from TF Hub on first run and caches locally
MODEL_URL = "https://tfhub.dev/tensorflow/efficientdet/d1/1"
CONFIDENCE_THRESHOLD = 0.3

# Show download progress in terminal during first run
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "1"

print("Loading EfficientDet model...")
model = hub.load(MODEL_URL)
print("EfficientDet loaded.")

def run_efficientdet(frame, gt_boxes=None):
    # Convert BGR (OpenCV format) to RGB (TensorFlow format)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Add batch dimension: shape becomes [1, H, W, 3]
    input_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)[tf.newaxis, ...]

    # Run inference
    detections = model(input_tensor)

    boxes = detections["detection_boxes"][0].numpy()
    scores = detections["detection_scores"][0].numpy()
    (h, w) = frame.shape[:2]

    pred_boxes, pred_scores = [], []
    for i in range(len(scores)):
        if scores[i] >= CONFIDENCE_THRESHOLD:
            y1, x1, y2, x2 = boxes[i]
            x1, x2 = int(x1 * w), int(x2 * w)
            y1, y2 = int(y1 * h), int(y2 * h)
            pred_boxes.append([x1, y1, x2, y2])
            pred_scores.append(float(scores[i]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{scores[i]:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    gt = gt_boxes if gt_boxes else []
    precision, recall, f1, mAP = compute_metrics(pred_boxes, pred_scores, gt)
    avg_confidence = round(float(sum(pred_scores) / len(pred_scores)), 4) if pred_scores else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP": mAP,
        "avg_confidence": avg_confidence,
        "num_detections": len(pred_boxes),
        "annotated_frame": frame,
    }