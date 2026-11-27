#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

int run_gguf(int argc, char** argv);

namespace {

constexpr std::size_t vocab_size = 256;
constexpr std::size_t group_size = 32;
constexpr std::size_t weight_count = vocab_size * vocab_size;
constexpr std::array<float, 8> e2m1{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

enum class Format { FP4, INT4 };

float decode_nibble(Format format, std::uint8_t code) {
    if (format == Format::FP4) return code & 8 ? -e2m1[code & 7] : e2m1[code];
    return static_cast<float>(code & 8 ? static_cast<int>(code) - 16 : code);
}

std::uint8_t encode_nibble(Format format, float normalized) {
    if (format == Format::INT4) {
        const auto code = static_cast<int>(std::round(std::clamp(normalized, -7.0f, 7.0f)));
        return static_cast<std::uint8_t>(code) & 0xf;
    }
    const float magnitude = std::abs(normalized);
    std::uint8_t nearest = 0;
    for (std::uint8_t i = 1; i < e2m1.size(); ++i) {
        if (std::abs(e2m1[i] - magnitude) < std::abs(e2m1[nearest] - magnitude)) nearest = i;
    }
    return normalized < 0.0f ? nearest | 8 : nearest;
}

struct Packed4 {
    explicit Packed4(Format fmt) : format(fmt), bytes(weight_count / 2), scales(weight_count / group_size) {}
    Format format;
    std::vector<std::uint8_t> bytes;
    std::vector<float> scales;

    float at(std::size_t i) const {
        const int shift = i % 2 == 0 ? 0 : 4;
        return decode_nibble(format, (bytes[i / 2] >> shift) & 0xf) * scales[i / group_size];
    }

    std::size_t storage_bytes() const {
        return bytes.size() + scales.size() * sizeof(float);
    }
};

std::string read_corpus(const char* path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error(std::string("cannot read corpus ") + path);
    std::string text(std::istreambuf_iterator<char>{input}, {});
    if (input.bad() || text.size() < 2) throw std::runtime_error("corpus must contain at least two bytes");
    return text;
}

std::vector<float> train(std::string_view corpus) {
    std::vector<float> matrix(weight_count, 0.125f);
    for (std::size_t i = 1; i < corpus.size(); ++i) {
        const auto prev = static_cast<unsigned char>(corpus[i - 1]);
        const auto next = static_cast<unsigned char>(corpus[i]);
        matrix[next * vocab_size + prev] += 1.0f;
    }
    for (float& cell : matrix) cell = std::log(cell);
    return matrix;
}

std::array<double, vocab_size> importance(std::string_view calibration) {
    // The expected squared one-hot feature equals its empirical frequency.
    std::array<double, vocab_size> counts{};
    counts.fill(0.05);
    for (std::size_t i = 1; i < calibration.size(); ++i) {
        ++counts[static_cast<unsigned char>(calibration[i - 1])];
    }
    return counts;
}

Packed4 quantize(const std::vector<float>& weights,
                 const std::array<double, vocab_size>& importance, Format format) {
    Packed4 output(format);
    for (std::size_t group = 0; group < output.scales.size(); ++group) {
        const auto start = group * group_size;
        float max_abs = 0.0f;
        for (std::size_t i = start; i < start + group_size; ++i) {
            max_abs = std::max(max_abs, std::abs(weights[i]));
        }
        const float max_number = format == Format::FP4 ? 6.0f : 7.0f;
        const float base_scale = max_abs == 0.0f ? 1.0f : max_abs / max_number;
        double best_error = INFINITY;
        float best_scale = base_scale;
        for (int percent = 50; percent <= 100; percent += 5) {
            const float scale = base_scale * (static_cast<float>(percent) / 100.0f);
            double error = 0.0;
            for (std::size_t i = start; i < start + group_size; ++i) {
                const auto code = encode_nibble(format, weights[i] / scale);
                const double diff = weights[i] - decode_nibble(format, code) * scale;
                error += importance[i % vocab_size] * diff * diff;
            }
            if (error < best_error) {
                best_error = error;
                best_scale = scale;
            }
        }
        output.scales[group] = best_scale;
        for (std::size_t i = start; i < start + group_size; i += 2) {
            const auto low = encode_nibble(format, weights[i] / best_scale);
            const auto high = encode_nibble(format, weights[i + 1] / best_scale);
            output.bytes[i / 2] = low | (high << 4);
        }
    }
    return output;
}

struct Metrics {
    double perplexity;
    double ns_per_prediction;
};

template <class Weight> Metrics evaluate(const Weight& weight, std::string_view data) {
    std::array<float, vocab_size> logits{};
    constexpr std::size_t repetitions = 50;
    const auto start = std::chrono::steady_clock::now();
    double nll = 0.0;
    for (std::size_t round = 0; round < repetitions; ++round) {
        for (std::size_t i = 1; i < data.size(); ++i) {
            const auto prev = static_cast<unsigned char>(data[i - 1]);
            const auto next = static_cast<unsigned char>(data[i]);
            for (std::size_t choice = 0; choice < vocab_size; ++choice) {
                logits[choice] = weight(choice * vocab_size + prev);
            }
            const double maximum = *std::max_element(logits.begin(), logits.end());
            double sum = 0.0;
            for (auto logit : logits) sum += std::exp(logit - maximum);
            nll += maximum + std::log(sum) - logits[next];
        }
    }
    const auto ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
    const auto n = static_cast<double>(repetitions * (data.size() - 1));
    return {std::exp(nll / n), ns / n};
}

void print_metrics(std::string_view label, std::size_t bytes, Metrics m, Metrics baseline) {
    std::cout << label << " bytes=" << bytes << " perplexity=" << m.perplexity
              << " delta_ppl=" << m.perplexity - baseline.perplexity
              << " delta_ppl_pct=" << 100.0 * (m.perplexity / baseline.perplexity - 1.0)
              << " ns_per_prediction=" << m.ns_per_prediction << '\n';
}

} // namespace

int main(int argc, char** argv) try {
    if (argc >= 2 && std::string_view(argv[1]) == "gguf") return run_gguf(argc, argv);
    if (argc != 1 && argc != 4) {
        throw std::invalid_argument("usage: quant_calibrate [train.txt calibrate.txt eval.txt] or quant_calibrate gguf ...");
    }
    const auto training = argc == 4 ? read_corpus(argv[1]) : std::string{
        "the river runs through the forest. the forest path follows the river."
        "\nthe sun warms the stone, and the wind cools the water. a boat crosses the river."
        "\na quiet afternoon turns to evening. the trees cast long shadows over the bank."
        "\na warm wind moves through the leaves, and the current carries a branch downstream."
        "\na small boat rests beside the shore. the river winds away through the hills.\n"
    };
    const auto calibration = argc == 4 ? read_corpus(argv[2]) : std::string{
        "the little path curves through the trees, and the wind moves across the water."
        "\nnear the shore, a quiet boat waits beside a stone. the hills are warm in the sun.\n"
    };
    const auto evaluation = argc == 4 ? read_corpus(argv[3]) : std::string{
        "evening settles over the forest. a boat drifts toward the bank as the wind slows."
        "\nthe quiet river carries leaves past the stone and into the hills.\n"
    };
    if (training == calibration || training == evaluation || calibration == evaluation) {
        throw std::invalid_argument("use distinct training, calibration, and evaluation corpora");
    }
    const auto weights = train(training);
    const auto activation_weights = importance(calibration);
    const auto baseline = evaluate([&](std::size_t i) { return weights[i]; }, evaluation);
    std::cout << std::fixed << std::setprecision(5)
              << "backend=cpu byte_model_charged_transitions=" << evaluation.size() - 1 << '\n';
    print_metrics("FP32", weights.size() * sizeof(float), baseline, baseline);
    for (const auto format : {Format::FP4, Format::INT4}) {
        const auto packed = quantize(weights, activation_weights, format);
        const auto metric = evaluate([&](std::size_t i) { return packed.at(i); }, evaluation);
        print_metrics(format == Format::FP4 ? "E2M1" : "INT4", packed.storage_bytes(), metric, baseline);
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
