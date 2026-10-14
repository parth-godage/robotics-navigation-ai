
# CPU vs GPU Benchmark

## Hardware
- GPU: NVIDIA GeForce RTX 2050 4GB
- Framework: PyTorch + Ultralytics YOLO
- Input: Single test image
- Benchmark: 5 warm-up runs, 30 measured runs

## Results

| Metric | CPU | RTX 2050 GPU |
|---|---:|---:|
| Average latency | 38.47 ms | 11.14 ms |
| Throughput | 25.99 FPS | 89.78 FPS |

GPU throughput was approximately 3.45x the CPU throughput
in this test.

Note: Results depend on hardware, model, image size,
software versions, and benchmark conditions.

