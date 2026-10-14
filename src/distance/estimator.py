class DistanceEstimator:
    def __init__(self, focal_length_px=800):
        self.focal_length_px = focal_length_px

        # Approximate assumed real-world heights (metres)
        self.object_heights = {
            "cone": 0.75,
            "barrier": 1.0,
            "stop sign": 0.75,
            "car": 1.5,
            "person": 1.7,
            "traffic light": 0.6,
        }

    def estimate(self, detection):
        class_name = detection["class_name"]
        x1, y1, x2, y2 = detection["bbox"]

        pixel_height = y2 - y1
        if pixel_height <= 0:
            return None

        real_height = self.object_heights.get(class_name)
        if real_height is None:
            return None

        distance = (
            self.focal_length_px * real_height / pixel_height
        )

        return round(distance, 2)