# FP4 and INT4 calibration

Compare FP4 and INT4 calibration on a byte model or benchmark quantized GGUF checkpoints.

## Run locally

Run from the repo root with CMake 3.24+, a C++20 compiler, and OpenSSL development files.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/quant_calibrate
```

[Calibration details, GGUF setup, and benchmarks](docs/reference.md)
