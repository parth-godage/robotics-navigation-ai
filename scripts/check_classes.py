
from pathlib import Path
from collections import Counter

label_dir = Path("D:/robotics-runs/cone-barrier-detection.v1-cone-barrier-v1.yolov11/train/labels")

counts = Counter()

for file in label_dir.glob("*.txt"):
    for line in file.read_text().splitlines():
        parts = line.split()
        if parts:
            counts[int(parts[0])] += 1

class_names = {
    0: "- water filled barrier - 2022-09-15 2-43pm",
    1: "pip",
    2: "water filled barrier - v9 2022-09-15 2-43pm",
}

print("Class distribution:")
for class_id, name in class_names.items():
    print(f"{class_id}: {name} -> {counts[class_id]} objects")

