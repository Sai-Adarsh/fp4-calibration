#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <regex>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef QUANT_HAS_NVML
#include <nvml.h>
#endif

extern char** environ;

namespace {

namespace fs = std::filesystem;
using Bytes = unsigned long long;

#ifdef QUANT_HAS_NVML
class GpuMemory {
  public:
    GpuMemory() : active_(nvmlInit_v2() == NVML_SUCCESS) {}
    ~GpuMemory() { if (active_) nvmlShutdown(); }
    bool active() const { return active_; }

    std::optional<Bytes> sample(pid_t pid) const {
        unsigned int devices = 0;
        if (nvmlDeviceGetCount_v2(&devices) != NVML_SUCCESS) return {};
        Bytes total = 0;
        bool observed = false;
        for (unsigned int index = 0; index < devices; ++index) {
            nvmlDevice_t device;
            if (nvmlDeviceGetHandleByIndex_v2(index, &device) != NVML_SUCCESS) return {};
            unsigned int count = 0;
            auto status = nvmlDeviceGetComputeRunningProcesses(device, &count, nullptr);
            if (status == NVML_SUCCESS) continue;
            if (status != NVML_ERROR_INSUFFICIENT_SIZE) return {};
            std::vector<nvmlProcessInfo_t> processes(count + 4);
            count = static_cast<unsigned int>(processes.size());
            status = nvmlDeviceGetComputeRunningProcesses(device, &count, processes.data());
            if (status != NVML_SUCCESS) return {};
            for (unsigned int i = 0; i < count; ++i) {
                if (processes[i].pid != static_cast<unsigned int>(pid)) continue;
                if (processes[i].usedGpuMemory == NVML_VALUE_NOT_AVAILABLE) return {};
                observed = true;
                total += processes[i].usedGpuMemory;
            }
        }
        if (!observed) return {};
        return total;
    }

  private:
    bool active_;
};
#endif

std::optional<Bytes> run(std::vector<std::string> args, const fs::path& output,
                         const fs::path& errors, bool memory = false) {
    posix_spawn_file_actions_t actions;
    int rc = posix_spawn_file_actions_init(&actions);
    if (rc != 0) throw std::runtime_error(std::strerror(rc));
    rc = posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, output.c_str(),
                                         O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (rc == 0) {
        rc = posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, errors.c_str(),
                                             O_WRONLY | O_CREAT | O_EXCL, 0600);
    }

    std::vector<char*> argv;
    for (auto& value : args) argv.push_back(value.data());
    argv.push_back(nullptr);
    pid_t pid = -1;
    if (rc == 0) rc = posix_spawn(&pid, argv.front(), &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (rc != 0) throw std::runtime_error(args.front() + ": " + std::strerror(rc));

    int status = 0;
    std::optional<Bytes> peak;
    bool finished = false;
#ifdef QUANT_HAS_NVML
    if (memory) {
        GpuMemory sampler;
        if (sampler.active()) {
            while (true) {
                const auto result = waitpid(pid, &status, WNOHANG);
                if (result == pid) {
                    finished = true;
                    break;
                }
                if (result < 0 && errno == EINTR) continue;
                if (result < 0) throw std::runtime_error(std::strerror(errno));
                if (const auto current = sampler.sample(pid)) {
                    peak = std::max(peak.value_or(0), *current);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
    }
#else
    (void) memory;
#endif
    if (!finished) {
        while (waitpid(pid, &status, 0) < 0) {
            if (errno != EINTR) throw std::runtime_error(std::strerror(errno));
        }
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        throw std::runtime_error(args.front() + " failed. See " + errors.string());
    }
    return peak;
}

std::optional<double> perplexity(const fs::path& filename) {
    const std::regex pattern(R"(Final estimate: PPL = ([0-9.]+))");
    std::ifstream input(filename);
    std::string line;
    std::smatch match;
    while (std::getline(input, line)) {
        if (std::regex_search(line, match, pattern)) return std::stod(match[1].str());
    }
    return {};
}

int parse_nonnegative(std::string_view text) {
    int value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value < 0) {
        throw std::invalid_argument("expected a nonnegative integer");
    }
    return value;
}

} // namespace

int run_gguf(int argc, char** argv) {
    if (argc < 7 || argc > 10) {
        throw std::invalid_argument("usage: quant_calibrate gguf BIN_DIR MODEL CALIBRATION EVALUATION NEW_OUTPUT_DIR [Q4_K_M|MXFP4_MOE] [GPU_LAYERS] [CONTEXT]");
    }
    const auto bin = fs::absolute(argv[2]);
    const auto model = fs::absolute(argv[3]);
    const auto calibration = fs::absolute(argv[4]);
    const auto evaluation = fs::absolute(argv[5]);
    const auto out = fs::absolute(argv[6]);
    const std::string format = argc >= 8 ? argv[7] : "Q4_K_M";
    const int layers = argc >= 9 ? parse_nonnegative(argv[8]) : 0;
    const int context = argc >= 10 ? parse_nonnegative(argv[9]) : 512;
    const auto layer_arg = std::to_string(layers);
    const auto context_arg = std::to_string(context);
    if (format != "Q4_K_M" && format != "MXFP4_MOE") throw std::invalid_argument("unsupported GGUF type");
    if (context < 32) throw std::invalid_argument("minimum context is 32");
    for (const auto& file : {model, calibration, evaluation}) {
        if (!fs::is_regular_file(file)) throw std::invalid_argument("input is not a regular file: " + file.string());
    }
    if (fs::equivalent(calibration, evaluation)) throw std::invalid_argument("use distinct calibration and evaluation files");
    for (const auto* name : {"llama-imatrix", "llama-quantize", "llama-perplexity", "llama-bench"}) {
        if (access((bin / name).c_str(), X_OK) != 0) throw std::runtime_error("missing executable: " + (bin / name).string());
    }
    if (!fs::create_directory(out)) throw std::invalid_argument("output directory already exists");

    const auto matrix = out / "imatrix.gguf";
    const auto packed = out / "quantized.gguf";
    run({(bin / "llama-imatrix").string(), "-m", model.string(), "-f", calibration.string(),
         "-o", matrix.string(), "-ngl", layer_arg, "-c", context_arg},
        out / "calibration.stdout", out / "calibration.stderr");
    run({(bin / "llama-quantize").string(), "--imatrix", matrix.string(), model.string(),
         packed.string(), format}, out / "quantize.stdout", out / "quantize.stderr");

    struct Measurement {
        double ppl;
        std::optional<Bytes> peak;
    };
    const auto measure = [&](const fs::path& checkpoint, std::string label) {
        auto stdout_path = out / (label + ".ppl.stdout");
        auto stderr_path = out / (label + ".ppl.stderr");
        run({(bin / "llama-perplexity").string(), "-m", checkpoint.string(), "-f", evaluation.string(),
             "-c", context_arg, "-ngl", layer_arg}, stdout_path, stderr_path);
        auto ppl = perplexity(stdout_path);
        if (!ppl) ppl = perplexity(stderr_path);
        if (!ppl || *ppl <= 0.0) throw std::runtime_error("missing final perplexity in " + label + " logs");
        auto peak = run({(bin / "llama-bench").string(), "-m", checkpoint.string(), "-ngl", layer_arg,
                         "-p", "512", "-n", "128", "-r", "5", "-o", "csv"},
                        out / (label + ".bench.csv"), out / (label + ".bench.stderr"), layers > 0);
        return Measurement{*ppl, peak};
    };

    const auto base = measure(model, "baseline");
    const auto low = measure(packed, "quantized");
    std::ofstream csv(out / "summary.csv");
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv << "checkpoint,gguf_bytes,perplexity,delta_ppl,delta_ppl_pct,sampled_peak_vram_bytes\n";
    const auto write_row = [&](std::string_view label, const fs::path& checkpoint, const Measurement& value) {
        csv << label << ',' << fs::file_size(checkpoint) << ',' << std::setprecision(12)
            << value.ppl << ',' << value.ppl - base.ppl << ','
            << 100.0 * (value.ppl / base.ppl - 1.0) << ',';
        if (value.peak) csv << *value.peak;
        csv << '\n';
    };
    write_row("baseline", model, base);
    write_row(format, packed, low);
    csv.close();
    std::cout << out / "summary.csv" << '\n';
    return 0;
}
