# Real-Time Object Detection Model Comparison

A benchmarking project that compares **YOLOv8**, **MobileNet-SSD**, and **EfficientDet** on live webcam feeds, evaluating real-time performance and detection quality metrics.

> **Authors:** Ayden Frometa & Payne Persons — Florida Polytechnic University, Department of Computer Science

---

## Project Overview

This project investigates which object detection model performs best under real-world webcam conditions. Each model represents a different design philosophy in real-time detection:

- **YOLOv8** — balanced accuracy and speed (Ultralytics)
- **MobileNet-SSD** — lightweight, resource-efficient baseline
- **EfficientDet** — accuracy-focused with compound scaling (BiFPN)

Models are evaluated on identical video footage to ensure a fair comparison.

---

## File Structure

```
CV_final_project/
├── main.py                         # Benchmark runner
├── yolov8_detector.py              # YOLOv8 inference wrapper
├── mobilenet_ssd_detector.py       # MobileNet-SSD inference wrapper
├── efficientdet_detector.py        # EfficientDet inference wrapper
├── record.py                       # Webcam video recorder
├── MobileNetSSD_deploy.prototxt    # MobileNet-SSD architecture file
├── MobileNetSSD_deploy.caffemodel  # MobileNet-SSD pre-trained weights (~23 MB)
├── webcam_test.mp4                 # Recorded webcam footage for benchmarking
├── results.csv                     # Benchmark output
└── utils/
    ├── __init__.py
    └── metrics.py                  # IoU, precision, recall, F1, mAP calculations
```

---

## Requirements

- Python 3.10
- Webcam (for live recording)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download MobileNet-SSD Model Files

```bash
curl -L -o MobileNetSSD_deploy.prototxt "https://raw.githubusercontent.com/Alina-chan/realtime-object-detection/master/MobileNetSSD_deploy.prototxt"
curl -L -o MobileNetSSD_deploy.caffemodel "https://raw.githubusercontent.com/Alina-chan/realtime-object-detection/master/MobileNetSSD_deploy.caffemodel"
```

Verify the `.caffemodel` is ~23 MB:
```powershell
(Get-Item MobileNetSSD_deploy.caffemodel).length / 1MB
```

> YOLOv8 weights (`yolov8n.pt`) download automatically on first run.
> EfficientDet downloads from TensorFlow Hub on first run.

---

## Usage

### Step 1 — Record Webcam Footage

```bash
python record.py
```

Press `Q` to stop recording. This saves `webcam_test.mp4` to your project folder.

**Tips for a good recording:**
- Walk toward and away from the camera
- Include common objects: bottle, chair, laptop, phone, cup
- Vary lighting conditions (turn lights on/off)
- Introduce motion blur by moving quickly
- Include multiple objects in the same frame
- Record at least 60 seconds of footage

### Step 2 — Run the Benchmark

```bash
python main.py
```

This runs all three models sequentially against `webcam_test.mp4` and saves results to `results.csv`.

**Frame counts per model (configured in `main.py`):**

| Model | Frames | Estimated Runtime |
|-------|--------|------------------|
| YOLOv8 | 900 | ~1.5 min |
| MobileNet-SSD | 900 | ~1.5 min |
| EfficientDet | 300 | ~8 min |

Press `Q` during any model's window to skip to the next model.

### Step 3 — View Results

Open `results.csv` — each row contains per-frame metrics for one model.

---

## Configuration

In `main.py`, adjust these variables at the top:

```python
VIDEO_SOURCE = "webcam_test.mp4"  # Change to 0 for live webcam

FRAME_COUNTS = {
    "YOLOv8": 900,
    "MobileNet-SSD": 900,
    "EfficientDet": 300,
}
```

---

## Output: results.csv

| Column | Description |
|--------|-------------|
| `model` | Model name |
| `frame_id` | Frame index |
| `fps` | Frames per second (inference speed) |
| `latency_ms` | Time to process one frame (ms) |
| `precision` | Detection precision (requires GT boxes) |
| `recall` | Detection recall (requires GT boxes) |
| `f1` | F1 score (requires GT boxes) |
| `mAP` | Mean Average Precision (requires GT boxes) |
| `avg_confidence` | Average detection confidence score per frame |
| `num_detections` | Number of objects detected per frame |

> **Note:** `precision`, `recall`, `f1`, and `mAP` require ground-truth bounding boxes to compute. When running on live/recorded webcam footage without annotations, these will be `0`. Use `avg_confidence` and `num_detections` as detection quality proxies in that case.

---

## Evaluation Metrics

### Real-Time Performance
- **FPS** — higher is better; minimum 24 FPS for smooth real-time use
- **Latency (ms)** — lower is better; under 42 ms for 24 FPS

### Detection Quality (with GT boxes)
- **Precision** — fraction of detections that are correct
- **Recall** — fraction of real objects detected
- **F1 Score** — harmonic mean of precision and recall
- **mAP (COCO)** — mean Average Precision using IoU ≥ 0.5 threshold

### Detection Quality (without GT boxes)
- **avg_confidence** — average model confidence across detections per frame
- **num_detections** — number of objects detected per frame

---

## Models

### YOLOv8
- Framework: Ultralytics
- Input size: variable (default 640×640)
- Weights: `yolov8n.pt` (nano variant)
- Classes: 80 COCO classes

### MobileNet-SSD
- Framework: OpenCV DNN module (Caffe)
- Input size: 300×300
- Weights: `MobileNetSSD_deploy.caffemodel`
- Classes: 20 VOC classes

### EfficientDet
- Framework: TensorFlow Hub
- Model: `efficientdet/d1/1`
- Input size: variable
- Classes: 90 COCO classes

---

## Known Issues

- **EfficientDet is slow** (~0.3–0.6 FPS on CPU) — not suitable for real-time use without a GPU
- **MobileNet-SSD** requires correctly paired `.prototxt` and `.caffemodel` files — mismatched versions cause OpenCV DNN errors
- Frame 0 of each model run shows inflated latency due to model warm-up — exclude it when computing summary statistics
