
from pathlib import Path
import shutil

# Original dataset path
DATASET = Path(
    r"D:\robotics-runs\cone-barrier-detection.v1-cone-barrier-v1.yolov11"
)

# Create a separate copy; original labels remain untouched
OUTPUT = DATASET.parent / "cone-barrier-merged"

if not (DATASET / "data.yaml").exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET}")

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Rename or remove it before rerunning."
    )

shutil.copytree(DATASET, OUTPUT)

# Merge class 2 (barrier) into class 0 (barrier)
# Class 1 (cone) remains unchanged
updated = 0

for split in ("train", "valid", "test"):
    labels_dir = OUTPUT / split / "labels"

    if not labels_dir.exists():
        print(f"Skipping missing folder: {labels_dir}")
        continue

    for label_file in labels_dir.glob("*.txt"):
        lines = label_file.read_text().splitlines()
        new_lines = []

        for line in lines:
            parts = line.split()
            if parts and parts[0] == "2":
                parts[0] = "0"
                updated += 1
            if parts:
                new_lines.append(" ".join(parts))

        label_file.write_text(
            "\n".join(new_lines) + ("\n" if new_lines else "")
        )

# Write corrected YAML with two classes and local paths
yaml_content = """path: .
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['barrier', 'cone']
"""

(OUTPUT / "data.yaml").write_text(yaml_content, encoding="utf-8")

print(f"\nDone! Updated {updated} barrier labels.")
print(f"New dataset: {OUTPUT}")
print("Original dataset was not modified.")
