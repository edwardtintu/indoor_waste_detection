import cv2
import numpy as np

# Load Pretrained MobileNet SSD Model
net = cv2.dnn.readNetFromCaffe("mobilenet_ssd.prototxt", "mobilenet_ssd.caffemodel")

# Define Class Labels
class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "dining table",
               "dog", "horse", "motorbike", "person", "potted plant",
               "sheep", "sofa", "train", "monitor"]

# Open Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert Frame to Blob
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop through detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])  # Class index
            label = class_names[idx] if idx < len(class_names) else "Unknown"
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, x_max, y_max) = box.astype("int")

            # Draw Bounding Box
            color = (0, 255, 0) if label != "person" else (0, 0, 255)  # Red if human
            cv2.rectangle(frame, (x, y), (x_max, y_max), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show Output
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
