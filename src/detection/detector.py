
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path="models/cone_barrier_best.pt"):
        # Custom-trained model: cones and barriers
        self.model = YOLO(model_path)

        # COCO model: stop signs
        self.stop_sign_model = YOLO("yolo11n.pt")

    def predict(self, image):
        detections = []

        # Detect cones and barriers
        results = self.model.predict(
            source=image,
            conf=0.25,
            iou=0.3,
            imgsz=640,
            device=0,
            verbose=False
        )

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                detections.append({
                    "class_name": self.model.names[class_id],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })

        # Detect stop signs using COCO class ID 11
        stop_results = self.stop_sign_model.predict(
            source=image,
            classes=[11],
            conf=0.25,
            imgsz=640,
            device=0,
            verbose=False
        )

        for result in stop_results:
            for box in result.boxes:
                detections.append({
                    "class_name": "stop sign",
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })

        return detections
