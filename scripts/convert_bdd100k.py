
import json
import random
import shutil
from pathlib import Path

# BDD100K source paths
SOURCE = Path(r"D:\archive (4)\bdd100k\bdd100k")
LABELS = Path(r"D:\archive (4)\bdd100k_labels_release\bdd100k\labels")

# Output dataset inside your project
OUTPUT = Path("data/bdd_yolo")

# Start with a small subset
TRAIN_LIMIT = 2000
VAL_LIMIT = 500

# BDD100K object-detection categories
CLASSES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]

CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def convert_split(split, limit):
    json_path = LABELS / f"bdd100k_labels_images_{split}.json"
    image_dir = SOURCE / "images" / "100k" / split
    output_images = OUTPUT / "images" / split
    output_labels = OUTPUT / "labels" / split

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        raise FileNotFoundError(f"Missing labels: {json_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing images: {image_dir}")

    with json_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    random.Random(42).shuffle(records)

    converted = 0
    for record in records:
        filename = record["name"]
        image_path = image_dir / filename

        if not image_path.exists():
            continue

        # Read dimensions from image to normalize boxes
        import cv2
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        height, width = image.shape[:2]
        lines = []

        for obj in record.get("labels", []):
            category = obj.get("category")
            box = obj.get("box2d")

            if category not in CLASS_TO_ID or not box:
                continue

            x1 = max(0.0, min(float(box["x1"]), width))
            y1 = max(0.0, min(float(box["y1"]), height))
            x2 = max(0.0, min(float(box["x2"]), width))
            y2 = max(0.0, min(float(box["y2"]), height))

            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            xc = ((x1 + x2) / 2) / width
            yc = ((y1 + y2) / 2) / height
            bw /= width
            bh /= height

            lines.append(
                f"{CLASS_TO_ID[category]} "
                f"{xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            )

        # Keep only images with usable detection labels
        if not lines:
            continue

        shutil.copy2(image_path, output_images / filename)
        label_path = output_labels / f"{Path(filename).stem}.txt"
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        converted += 1
        if converted >= limit:
            break

    print(f"{split}: converted {converted} images")


if __name__ == "__main__":
    convert_split("train", TRAIN_LIMIT)
    convert_split("val", VAL_LIMIT)
