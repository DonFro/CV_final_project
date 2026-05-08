import cv2
import numpy as np
from utils.metrics import compute_metrics

# Paths to the Caffe model files — must be correctly paired versions
# OpenCV reads the prototxt first to build the network structure in memory, then loads the caffemodel to fill every layer with its trained weights. After that the network is ready to run inference.
PROTOTXT = "MobileNetSSD_deploy.prototxt"
CAFFEMODEL = "MobileNetSSD_deploy.caffemodel"
CONFIDENCE_THRESHOLD = 0.2

# MobileNet-SSD was trained on Pascal VOC — 20 object classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Load the network from Caffe model files once at import time
net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

def run_mobilenet_ssd(frame, gt_boxes=None):
    (h, w) = frame.shape[:2]
    # Preprocess: resize to 300x300, normalize pixel values to [-1, 1]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                  0.007843, (300, 300), 127.5)
    net.setInput(blob)

    # Run forward pass to get detections
    detections = net.forward()

    pred_boxes, pred_scores = [], []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections below the confidence threshold
        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            pred_boxes.append([x1, y1, x2, y2])
            pred_scores.append(float(confidence))
            label = f"{CLASSES[int(detections[0,0,i,1])]}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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