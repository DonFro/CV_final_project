import csv
import time
import cv2
import psutil

from yolov8_detector import run_yolov8
# from mobilenet_ssd_detector import run_mobilenet_ssd
from efficientdet_detector import run_efficientdet

MODELS = {
    "YOLOv8": run_yolov8,
    # "MobileNet-SSD": run_mobilenet_ssd,
    "EfficientDet": run_efficientdet,
}

LOG_FILE = "results.csv"

def log_result(model_name, frame_id, fps, latency_ms, precision, recall, f1, mAP):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([model_name, frame_id, fps, latency_ms, precision, recall, f1, mAP])

def init_log():
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "frame_id", "fps", "latency_ms", "precision", "recall", "f1", "mAP"])

def run_benchmark(model_name, detect_fn, num_frames=300):
    cap = cv2.VideoCapture(0)
    print(f"\n--- Running {model_name} ---")

    for frame_id in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        detections = detect_fn(frame)  # returns dict with precision, recall, f1, mAP
        latency_ms = (time.time() - start) * 1000
        fps = 1000 / latency_ms if latency_ms > 0 else 0

        log_result(
            model_name, frame_id, round(fps, 2), round(latency_ms, 2),
            detections.get("precision", 0),
            detections.get("recall", 0),
            detections.get("f1", 0),
            detections.get("mAP", 0),
        )

        cv2.imshow(model_name, detections.get("annotated_frame", frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"{model_name} done. Results saved to {LOG_FILE}")

if __name__ == "__main__":
    init_log()
    for name, fn in MODELS.items():
        run_benchmark(name, fn, num_frames=300)
    print("\nAll models benchmarked. Open results.csv to compare.")