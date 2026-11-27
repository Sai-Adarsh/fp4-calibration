#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    const auto tool = std::filesystem::path(argv[0]).filename().string();
    const auto flag = [&](std::string_view name) -> std::string_view {
        for (int i = 1; i + 1 < argc; ++i) {
            if (argv[i] == name) return argv[i + 1];
        }
        return {};
    };
    const auto write = [](std::string_view path, std::string_view bytes) {
        std::ofstream output{std::filesystem::path(path), std::ios::binary};
        output << bytes;
        return output.good() ? 0 : 9;
    };
    const auto getenv = [](const char* name) -> std::string_view {
        const char* value = std::getenv(name);
        return value ? value : "";
    };

    if (tool == "llama-imatrix") {
        const auto corpus = std::filesystem::path(flag("-f")).filename().string();
        if (corpus != "calibration.txt" || flag("-c") != flag("-b")) return 8;
        return write(flag("-o"), "fake importance matrix");
    }
    if (tool == "llama-quantize") {
        if (argc != 6 || std::string_view(argv[1]) != "--imatrix"
            || !std::filesystem::exists(argv[2])) return 8;
        return write(argv[4], "GGUF fake Q4 payload");
    }
    if (tool == "llama-perplexity") {
        if (getenv("QUANT_TEST_MODE") == "crash") {
            std::cerr << "simulated subprocess failure\n";
            return 7;
        }
        const auto corpus = std::filesystem::path(flag("-f")).filename().string();
        if (corpus != "evaluation.txt" || flag("-c") != flag("-b")) return 8;
        if (getenv("QUANT_TEST_MODE") == "missing_ppl") {
            std::cerr << "Final estimate: PPL = NaN\n";
        } else if (std::filesystem::path(flag("-m")).filename() == "quantized.gguf") {
            std::cerr << "Final estimate: PPL = 1.05e+2 +/- 1.75\n";
        } else {
            std::cout << "Final estimate: PPL = 100\n";
        }
        return 0;
    }
    if (tool == "llama-bench") {
        if (flag("-p") != "128" || flag("-n") != "128" || flag("-r") != "5") return 8;
        std::cout << "n_prompt,n_gen,avg_ts\n128,0,1000\n0,128,50\n";
        return 0;
    }
    return 9;
}
