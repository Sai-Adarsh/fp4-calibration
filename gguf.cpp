#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <regex>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <openssl/evp.h>
#ifdef QUANT_HAS_NVML
#include <nvml.h>
#endif

extern char** environ;

namespace {

namespace fs = std::filesystem;
using Bytes = unsigned long long;

struct Fingerprint {
    Bytes bytes;
    std::string sha256;

    bool operator==(const Fingerprint&) const = default;
};

struct Artifact {
    std::string_view role;
    fs::path path;
    Fingerprint identity;
};

Fingerprint fingerprint(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> hash(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!hash || EVP_DigestInit_ex(hash.get(), EVP_sha256(), nullptr) != 1) {
        throw std::runtime_error("cannot initialize SHA-256");
    }
    std::vector<char> buffer(1024 * 1024);
    Bytes bytes = 0;
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto size = static_cast<std::size_t>(input.gcount());
        bytes += size;
        if (size != 0 && EVP_DigestUpdate(hash.get(), buffer.data(), size) != 1) {
            throw std::runtime_error("cannot hash " + path.string());
        }
    }
    if (input.bad() || !input.eof()) throw std::runtime_error("cannot read " + path.string());
    std::array<unsigned char, 32> digest{};
    unsigned int length = 0;
    if (EVP_DigestFinal_ex(hash.get(), digest.data(), &length) != 1 || length != digest.size()) {
        throw std::runtime_error("cannot finalize SHA-256");
    }
    constexpr std::string_view hex = "0123456789abcdef";
    std::string encoded;
    encoded.reserve(2 * digest.size());
    for (auto value : digest) {
        encoded.push_back(hex[value >> 4]);
        encoded.push_back(hex[value & 15]);
    }
    return {bytes, std::move(encoded)};
}

void manifest_row(std::ostream& manifest, std::string_view role, const fs::path& file,
                  const Fingerprint& identity) {
    manifest << role << ",\"";
    for (const char c : file.filename().string()) {
        if (c == '"') manifest << '"';
        manifest << c;
    }
    manifest << "\"," << identity.bytes << ',' << identity.sha256 << '\n';
}

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

struct Perplexity {
    double value;
    std::optional<double> reported_error;
};

std::optional<Perplexity> perplexity(const fs::path& filename) {
    const std::string number = R"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)";
    const std::regex pattern("Final estimate: PPL =\\s*(" + number + ")(?:\\s*\\+/-\\s*(" + number + "))?\\s*$");
    std::ifstream input(filename);
    std::string line;
    std::smatch match;
    while (std::getline(input, line)) {
        if (!std::regex_search(line, match, pattern)) continue;
        const double value = std::stod(match[1].str());
        if (!std::isfinite(value) || value <= 0) continue;
        std::optional<double> error;
        if (match[2].matched) {
            const double parsed = std::stod(match[2].str());
            if (!std::isfinite(parsed) || parsed < 0) continue;
            error = parsed;
        }
        return Perplexity{value, error};
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

double parse_budget(const char* text) {
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value) || value < 0) {
        throw std::invalid_argument("PPL increase budget must be finite and nonnegative");
    }
    return value;
}

} // namespace

int run_gguf(int argc, char** argv) {
    if (argc < 7 || argc > 11) {
        throw std::invalid_argument("usage: quant_calibrate gguf BIN_DIR MODEL CALIBRATION EVALUATION NEW_OUTPUT_DIR [Q4_K_M|MXFP4_MOE] [GPU_LAYERS] [CONTEXT] [MAX_PPL_INCREASE_PCT]");
    }
    const auto bin = fs::absolute(argv[2]);
    const auto model = fs::absolute(argv[3]);
    const auto calibration = fs::absolute(argv[4]);
    const auto evaluation = fs::absolute(argv[5]);
    const auto out = fs::absolute(argv[6]);
    const std::string format = argc >= 8 ? argv[7] : "Q4_K_M";
    const int layers = argc >= 9 ? parse_nonnegative(argv[8]) : 0;
    const int context = argc >= 10 ? parse_nonnegative(argv[9]) : 512;
    const std::optional<double> budget = argc == 11 ? std::optional{parse_budget(argv[10])} : std::nullopt;
    const auto layer_arg = std::to_string(layers);
    const auto context_arg = std::to_string(context);
    const auto ubatch_arg = std::to_string(std::min(context, 512));
    const auto prefill_arg = std::to_string(std::min(context, 512));
    const auto decode_arg = std::to_string(std::min(context, 128));
    if (format != "Q4_K_M" && format != "MXFP4_MOE") throw std::invalid_argument("unsupported GGUF type");
    if (context < 32) throw std::invalid_argument("minimum context is 32");
    if (std::regex_search(model.filename().string(), std::regex(R"(-[0-9]{5}-of-[0-9]{5}\.gguf$)"))) {
        throw std::invalid_argument("the manifest requires a single-file GGUF checkpoint");
    }
    for (const auto& file : {model, calibration, evaluation}) {
        if (!fs::is_regular_file(file)) throw std::invalid_argument("input is not a regular file: " + file.string());
    }
    const auto calibration_id = fingerprint(calibration);
    const auto evaluation_id = fingerprint(evaluation);
    if (calibration_id.bytes == 0 || evaluation_id.bytes == 0) {
        throw std::invalid_argument("calibration and evaluation texts must be nonempty");
    }
    if (calibration_id.sha256 == evaluation_id.sha256) {
        throw std::invalid_argument("calibration and evaluation contents have the same SHA-256");
    }
    const auto model_id = fingerprint(model);
    constexpr std::array tools{"llama-imatrix", "llama-quantize", "llama-perplexity", "llama-bench"};
    for (const auto* name : tools) {
        if (access((bin / name).c_str(), X_OK) != 0) throw std::runtime_error("missing executable: " + (bin / name).string());
    }
    if (!fs::create_directory(out)) throw std::invalid_argument("output directory already exists");

    std::vector<Artifact> artifacts;
    std::ofstream manifest(out / "manifest.csv");
    manifest.exceptions(std::ios::badbit | std::ios::failbit);
    manifest << "role,filename,bytes,sha256\n";
    const auto record = [&](std::string_view role, const fs::path& path, const Fingerprint& identity) {
        manifest_row(manifest, role, path, identity);
        artifacts.push_back({role, path, identity});
    };
    record("baseline", model, model_id);
    record("calibration", calibration, calibration_id);
    record("evaluation", evaluation, evaluation_id);
    for (const auto* name : tools) {
        const auto executable = bin / name;
        record("tool", executable, fingerprint(executable));
    }
    manifest.flush();

    std::ofstream commands(out / "commands.txt");
    commands.exceptions(std::ios::badbit | std::ios::failbit);
    const auto execute = [&](std::vector<std::string> args, const fs::path& stdout_path,
                             const fs::path& stderr_path, bool memory = false) {
        for (const auto& arg : args) commands << std::quoted(arg) << ' ';
        commands << '\n' << std::flush;
        return run(std::move(args), stdout_path, stderr_path, memory);
    };

    const auto matrix = out / "imatrix.gguf";
    const auto packed = out / "quantized.gguf";
    execute({(bin / "llama-imatrix").string(), "-m", model.string(), "-f", calibration.string(),
         "-o", matrix.string(), "-ngl", layer_arg, "-c", context_arg, "-b", context_arg, "-ub", ubatch_arg},
        out / "calibration.stdout", out / "calibration.stderr");
    record("importance_matrix", matrix, fingerprint(matrix));
    execute({(bin / "llama-quantize").string(), "--imatrix", matrix.string(), model.string(),
         packed.string(), format}, out / "quantize.stdout", out / "quantize.stderr");

    const auto packed_id = fingerprint(packed);
    record("quantized", packed, packed_id);
    manifest.close();

    struct Measurement {
        Perplexity ppl;
        std::optional<Bytes> peak;
    };
    const auto measure = [&](const fs::path& checkpoint, std::string label) {
        auto stdout_path = out / (label + ".ppl.stdout");
        auto stderr_path = out / (label + ".ppl.stderr");
        execute({(bin / "llama-perplexity").string(), "-m", checkpoint.string(), "-f", evaluation.string(),
             "-c", context_arg, "-b", context_arg, "-ub", ubatch_arg, "-ngl", layer_arg}, stdout_path, stderr_path);
        auto ppl = perplexity(stdout_path);
        if (!ppl) ppl = perplexity(stderr_path);
        if (!ppl) throw std::runtime_error("missing final perplexity in " + label + " logs");
        auto peak = execute({(bin / "llama-bench").string(), "-m", checkpoint.string(), "-ngl", layer_arg,
                         "-p", prefill_arg, "-n", decode_arg, "-r", "5", "-o", "csv"},
                        out / (label + ".bench.csv"), out / (label + ".bench.stderr"), layers > 0);
        return Measurement{*ppl, peak};
    };

    const auto base = measure(model, "baseline");
    const auto low = measure(packed, "quantized");
    for (const auto& artifact : artifacts) {
        if (!fs::is_regular_file(artifact.path) || fingerprint(artifact.path) != artifact.identity) {
            throw std::runtime_error("artifact changed during run: " + std::string(artifact.role) +
                                     " (" + artifact.path.string() + ")");
        }
    }
    std::ofstream csv(out / "summary.csv");
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv << "checkpoint,gguf_bytes,perplexity,llama_ppl_reported_error,delta_ppl,delta_ppl_pct,"
           "delta_nll_nats_per_token,sampled_peak_vram_bytes\n";
    const auto write_row = [&](std::string_view label, const Fingerprint& checkpoint, const Measurement& value) {
        const double ratio = value.ppl.value / base.ppl.value;
        csv << label << ',' << checkpoint.bytes << ',' << std::setprecision(9) << value.ppl.value << ',';
        if (value.ppl.reported_error) csv << *value.ppl.reported_error;
        csv << ',' << value.ppl.value - base.ppl.value << ',' << 100.0 * (ratio - 1.0)
            << ',' << std::log(ratio) << ',';
        if (value.peak) csv << *value.peak;
        csv << '\n';
    };
    write_row("baseline", model_id, base);
    write_row(format, packed_id, low);
    csv.close();
    std::cout << out / "summary.csv" << '\n';
    const double increase = 100.0 * (low.ppl.value / base.ppl.value - 1.0);
    if (budget) {
        std::ofstream gate(out / "quality-gate.csv");
        gate.exceptions(std::ios::badbit | std::ios::failbit);
        gate << "max_ppl_increase_pct,observed_ppl_increase_pct,status\n";
        gate << std::setprecision(9) << *budget << ',' << increase << ',' << (increase > *budget ? "fail" : "pass") << '\n';
        gate.close();
        if (increase > *budget) {
            std::cerr << "perplexity increase " << increase << "% exceeds " << *budget << "% budget\n";
            return 3;
        }
    }
    return 0;
}
