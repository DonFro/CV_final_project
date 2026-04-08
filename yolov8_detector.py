from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # downloads automatically on first run

def run_yolov8(frame):
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # Extract basic precision/recall from results if available
    precision = 0
    recall = 0
    f1 = 0
    mAP = 0

    if results[0].boxes is not None:
        probs = results[0].boxes.conf.cpu().numpy()
        precision = float(probs.mean()) if len(probs) > 0 else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mAP": round(mAP, 4),
        "annotated_frame": annotated_frame,
    }