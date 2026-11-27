# FP4 and INT4 calibration

The C++ runner requires CMake 3.24 or later and OpenSSL libcrypto for SHA-256.

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

Use a single-file, high-precision GGUF checkpoint, separate calibration and evaluation texts, and a llama.cpp build with `llama-imatrix`, `llama-quantize`, `llama-perplexity`, and `llama-bench`. Give the driver a new output directory with room for the quantized checkpoint and importance matrix.

```sh
./build/quant_calibrate gguf /path/to/llama.cpp/build/bin /path/to/model-f16.gguf /path/to/calibration.txt /path/to/eval.txt ./q4-run Q4_K_M 0 512 5
```

The optional arguments select the format, GPU layer count, context size, and maximum percentage increase in perplexity. The quality gate writes its decision and exits with status 3 when the point estimate exceeds the budget. It preserves the measurements. For compatible MoE checkpoints, use `MXFP4_MOE` to quantize the expert tensors.

Calibration and evaluation run with the requested context as the logical batch size. The runner refuses identical corpus SHA-256 hashes, including renamed copies. Corpus selection still determines how representative the evaluation is. The benchmark measures prefill up to 512 tokens and decode up to 128 tokens, each capped at the requested context with five repetitions.

- `manifest.csv` records file sizes and SHA-256 hashes for the checkpoint, corpora, tool executables, importance matrix, and output checkpoint.
- `commands.txt` and phase logs capture tool invocations and output.
- `summary.csv` reports held-out perplexity, any error estimate printed by llama.cpp, percentage change, and the difference in negative log likelihood in nats per token.
- `*.bench.csv` preserves llama.cpp throughput and timing. For NVIDIA memory samples, build with `-DQUANT_ENABLE_NVML=ON` and use a nonzero GPU layer count. The 50 ms sampling interval gives a lower bound on the peak.

The runner hashes the inputs before the tools start and the generated files as soon as their producing phase finishes. It rechecks those fingerprints after the measurements and before writing the summary or quality decision. If a file's content changes, the run fails and preserves the phase logs. Keep the inputs stationary during a run. The two checks detect persistent content changes and add a second read of each checkpoint outside the measured phases.

## Measured TinyStories run

An Apple M3 Pro ran llama.cpp `b12810` / `ca123ba76` with zero model layers offloaded. The [TinyStories checkpoint](https://huggingface.co/ggml-org/models-moved/blob/main/tinyllamas/stories15M.gguf) contained 24,407,712 parameters according to llama.cpp. Calibration used 36 chunks and evaluation used 35 chunks, both with 128 tokens per context and one sequence per batch.

| Checkpoint | GGUF bytes | Held-out PPL | Prefill tok/s ± SD | Decode tok/s ± SD |
|---|---:|---:|---:|---:|
| F32 | 98,357,920 | 1,335.762 | 37,761 ± 7,211 | 1,280 ± 110 |
| Q4_K_M | 20,986,432 | 1,346.235 | 23,831 ± 2,395 | 4,701 ± 473 |

On this run Q4_K_M shrank the file by 4.69×, raised measured perplexity by 0.784%, and decoded at 3.67× the F32 rate. Prefill slowed. The tool reported PPL errors around 137 for both variants, larger than the observed change. The quality gate's 25% budget is a sample configuration. No VRAM measurements were taken.

To reproduce the input texts, take 16,000 bytes starting at offset 4,096 from [Alice's Adventures in Wonderland](https://github.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/blob/7f18748e114b994d5d9454d5793581bc0b9cf9d7/11.txt) for calibration and [Pride and Prejudice](https://github.com/GITenberg/Pride-and-Prejudice_1342/blob/81db45c9c48c592f0b77f01fc59e677ad0a5634e/1342.txt) for evaluation. Run with `Q4_K_M 0 128 25` and use the [manifest](evidence/tinystories-manifest.csv) to match input hashes. See the [quality](evidence/tinystories-quality.csv) and [throughput](evidence/tinystories-throughput.csv) measurements. This experiment covers one small checkpoint and a narrow evaluation corpus.

[llama.cpp importance matrix](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md) · [Quantizer](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) · [Perplexity](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md) · [Benchmarking](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md)

## Tests and coverage

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

The suite drives the public CLI for both the byte model and a temporary GGUF pipeline. Local test binaries stand in for the four llama.cpp tools. The tests check that separate inputs reach calibration and evaluation, compare recorded hashes with CMake's SHA-256, and check the quality gate. A fixture that changes the corpus or packed checkpoint during the run must leave the phase logs without producing a summary. The tests require neither a checkpoint download nor a GPU.

For line and branch coverage on macOS with Xcode:

```sh
cov_dir=$(mktemp -d)
mkdir "$cov_dir/raw"
cmake -S . -B "$cov_dir/build" -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping" \
  -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-generate"
cmake --build "$cov_dir/build"
LLVM_PROFILE_FILE="$cov_dir/raw/%m-%p.profraw" ctest --test-dir "$cov_dir/build" --output-on-failure
xcrun llvm-profdata merge -sparse "$cov_dir"/raw/*.profraw -o "$cov_dir/merged.profdata"
xcrun llvm-cov report "$cov_dir/build/quant_calibrate" -instr-profile "$cov_dir/merged.profdata"
```

The Apple Silicon build covered 415 of 423 compiled production lines (98.11%) and 175 of 208 branches (84.13%). The uncovered lines handle failures from OpenSSL and `waitpid`. The optional NVML code requires a separate NVIDIA build and measurement.
