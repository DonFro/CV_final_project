import cv2
import os

OUTPUT_FILE = "webcam_test.mp4"
FPS = 30
RESOLUTION = (640, 480)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, RESOLUTION)

print("Recording... Press Q to stop.")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    out.write(frame)
    frame_count += 1

    # Show frame count on screen
    display = frame.copy()
    cv2.putText(display, f"Recording... Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(display, "Press Q to stop", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Recording", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

duration_sec = frame_count / FPS
print(f"\nSaved to {OUTPUT_FILE}")
print(f"Frames recorded: {frame_count}")
print(f"Duration: {duration_sec:.1f} seconds")
print(f"Sufficient for: YOLOv8/MobileNet-SSD (900 frames = {900/FPS:.0f}s needed), EfficientDet (300 frames = {300/FPS:.0f}s needed)")