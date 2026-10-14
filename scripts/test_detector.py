
import cv2

from src.detection.detector import ObjectDetector
from src.distance.estimator import DistanceEstimator
from src.visualization.annotator import draw_detections


IMAGE_PATH = "test.png"
OUTPUT_PATH = "outputs/robotics_result.jpg"


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    detector = ObjectDetector()
    estimator = DistanceEstimator()

    detections = detector.predict(image)

    for item in detections:
        item["distance_m"] = estimator.estimate(item)

        print(
            f"Class: {item['class_name']} | "
            f"Confidence: {item['confidence']:.2f} | "
            f"Distance: {item['distance_m']} m"
        )

    output = draw_detections(image, detections)

    if not cv2.imwrite(OUTPUT_PATH, output):
        raise RuntimeError("Could not save output image")

    print(f"\nTotal detections: {len(detections)}")
    print(f"Saved result: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
