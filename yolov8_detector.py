from ultralytics import YOLO
from utils.metrics import compute_metrics

model = YOLO("yolov8n.pt")

def run_yolov8(frame, gt_boxes=None):
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    pred_boxes, pred_scores = [], []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            pred_boxes.append([x1, y1, x2, y2])
            pred_scores.append(float(box.conf[0].cpu().numpy()))

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
        "annotated_frame": annotated_frame,
    }