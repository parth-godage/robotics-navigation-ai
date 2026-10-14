
import time
import cv2
import torch
from ultralytics import YOLO

IMAGE_PATH = "test.png"
MODEL_PATH = "models/cone_barrier_best.pt"
RUNS = 30

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

model = YOLO(MODEL_PATH)


def benchmark(device):
    # Warm-up
    for _ in range(5):
        model.predict(image, device=device, verbose=False)

    if device == 0:
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(RUNS):
        model.predict(image, device=device, verbose=False)

    if device == 0:
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    avg_latency = elapsed / RUNS
    fps = RUNS / elapsed

    return avg_latency * 1000, fps


for device in ["cpu", 0]:
    try:
        latency, fps = benchmark(device)
        name = "CPU" if device == "cpu" else "RTX 2050 GPU"

        print(f"\n{name}")
        print(f"Average latency: {latency:.2f} ms")
        print(f"FPS: {fps:.2f}")

    except Exception as error:
        print(f"Benchmark failed for {device}: {error}")
