
import cv2


def draw_detections(image, detections):
    output = image.copy()

    for item in detections:
        x1, y1, x2, y2 = map(int, item["bbox"])

        distance = item.get("distance_m")

        if distance is not None:
            label = (
                f'{item["class_name"]} '
                f'{distance:.2f} m'
            )
        else:
            label = item["class_name"]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    return output
