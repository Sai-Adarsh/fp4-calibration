# FP4 and INT4 calibration

## What it is

Compare FP4 and INT4 calibration on a byte model or benchmark quantized GGUF checkpoints.

## How to run

Run from the repo root with CMake 3.24+, a C++20 compiler, and OpenSSL development files.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/quant_calibrate
```

## Other things

[Calibration details, GGUF setup, and benchmarks](docs/reference.md)
