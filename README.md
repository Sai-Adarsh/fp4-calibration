# FP4 and INT4 calibration

## Byte model

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/quant_calibrate
```

The CPU program trains a smoothed byte-level bigram model. On a separate calibration corpus, it estimates input activation frequencies and chooses an FP32 scale for each group of 32 weights by minimizing weighted reconstruction error over candidate clipping levels. It stores two E2M1 or symmetric INT4 codes per byte. INT4 codes range from −7 to +7.

The evaluator reads weights from the packed representation and reports storage including scales, held-out perplexity and its change from FP32, and CPU time per next-byte prediction including normalization. To use your own data, pass distinct training, calibration, and evaluation corpus paths in that order.

The E2M1 reference uses FP32 group scales. The NVFP4 format uses an FP8 scale per 16 values and a separate tensor scale. [NVIDIA NVFP4 format](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)

## GGUF checkpoints

Use a high-precision GGUF checkpoint, separate calibration and evaluation corpora, and a llama.cpp build with `llama-imatrix`, `llama-quantize`, `llama-perplexity`, and `llama-bench`. Give the driver a new output directory with room for the quantized checkpoint and importance matrix.

```sh
./build/quant_calibrate gguf /path/to/llama.cpp/build/bin /path/to/model-f16.gguf /path/to/calibration.txt /path/to/eval.txt ./q4-run Q4_K_M 0 512
```

The last three arguments set the quantization type, GPU layers, and perplexity context. For compatible MoE checkpoints, run `MXFP4_MOE` in a separate directory to quantize expert tensors.

`summary.csv` compares baseline and quantized file size and perplexity at llama.cpp's reported precision. The `*.bench.csv` files report prompt processing and decode times, and the phase logs preserve tool output. On NVIDIA systems, build with `-DQUANT_ENABLE_NVML=ON` and pass a nonzero GPU layer count to sample benchmark process memory across devices every 50 ms. The sampled peak is a lower bound. The VRAM field is empty when no sample is available.

[llama.cpp importance matrix](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md) · [Quantizer](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) · [Perplexity](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md) · [Benchmarking](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md)
