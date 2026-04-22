// Qwen2.5-VL-7B interactive chat example.
//

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "geniex-proc/qwen2vl.h"
#include "geniex-proc/tokenizer.h"
#include "qwen2_5_vl.h"
#include "types.h"
#include "vlm/vlm_types.h"

#ifdef _WIN32
#include <windows.h>
static void enable_utf8_io() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    DWORD  mode = 0;
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (GetConsoleMode(hOut, &mode))
        SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}
#endif

// ── Argument parsing ──────────────────────────────────────────────────────────

struct Args {
    int32_t     max_tokens    = 512;
    bool        verbose       = false;
    std::string system_prompt = "You are a helpful AI assistant";
};

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --max-tokens <n>      Max tokens to generate (default 512)\n"
              << "  --system-prompt <s>   System prompt\n"
              << "  --verbose             Print performance metrics\n"
              << "  --help\n"
              << "\n"
              << "At the prompt, include an image path anywhere in the line:\n"
              << "  > describe this picture /path/to/cat.jpg\n";
}

static bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : std::string{};
        };
        if      (a == "--max-tokens")    args.max_tokens    = std::stoi(next());
        else if (a == "--system-prompt") args.system_prompt = next();
        else if (a == "--verbose")       args.verbose       = true;
        else if (a == "--help" || a == "-h") { printUsage(argv[0]); return false; }
        else { std::cerr << "Unknown argument: " << a << "\n"; return false; }
    }
    return true;
}

// ── File type detection ───────────────────────────────────────────────────────

static bool isImageFile(const std::string& path) {
    std::string p = path;
    std::transform(p.begin(), p.end(), p.begin(), ::tolower);
    return p.ends_with(".jpg") || p.ends_with(".jpeg") || p.ends_with(".png")
        || p.ends_with(".bmp") || p.ends_with(".gif")  || p.ends_with(".webp");
}

// Splits an input line into text + image paths.
static void parseInput(const std::string& input,
                       std::string& prompt_text,
                       std::vector<std::string>& image_paths) {
    image_paths.clear();
    std::vector<std::string> text_tokens;
    std::istringstream iss(input);
    std::string token;
    while (iss >> token) {
        if (isImageFile(token)) image_paths.push_back(token);
        else                    text_tokens.push_back(token);
    }
    prompt_text.clear();
    for (size_t i = 0; i < text_tokens.size(); ++i) {
        if (i > 0) prompt_text += ' ';
        prompt_text += text_tokens[i];
    }
}

// BatchFeatures (xtensor) → PixelData (flat), as consumed by the VLM layer.
static geniex::PixelData toPixelData(const geniex::BatchFeatures& bf) {
    geniex::PixelData pd;
    if (bf.image_grid_thw.dimension() == 0 || bf.image_grid_thw.shape()[0] == 0) return pd;

    pd.pixel_values.assign(bf.pixel_values.cbegin(), bf.pixel_values.cend());

    const size_t n = bf.image_grid_thw.shape()[0];
    pd.image_grid_thw.resize(n);
    for (size_t i = 0; i < n; ++i) {
        pd.image_grid_thw[i] = {
            static_cast<int32_t>(bf.image_grid_thw(i, 0)),
            static_cast<int32_t>(bf.image_grid_thw(i, 1)),
            static_cast<int32_t>(bf.image_grid_thw(i, 2)),
        };
    }
    return pd;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
#ifdef _WIN32
    enable_utf8_io();
#endif

    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    const auto model_dir =
        std::filesystem::current_path() / "modelfiles" / "qwen2_5_vl_7b";

    // All QNN runtime paths are left as std::nullopt → auto-detected.
    geniex::QnnRuntimeConfig runtime_cfg;

    // ── Model config ─────────────────────────────────────────────────────────

    geniex::qwen2_5_vl_7b::Qwen25VLConfig config;

    config.llm_config.model_paths = {
        (model_dir / "part1_of_5.bin").string(),
        (model_dir / "part2_of_5.bin").string(),
        (model_dir / "part3_of_5.bin").string(),
        (model_dir / "part4_of_5.bin").string(),
        (model_dir / "part5_of_5.bin").string(),
    };
    config.llm_config.tokenizer_path  = (model_dir / "tokenizer.json").string();
    config.llm_config.htp_config_path = (model_dir / "htp_backend_ext_config.json").string();
    config.llm_config.embedding_path  = (model_dir / "embedding_weights.raw").string();

    config.vision_config.model_paths      = {(model_dir / "vision_encoder.bin").string()};
    config.vision_config.htp_config_path  = (model_dir / "htp_backend_ext_config.json").string();

    geniex::GenerationConfig gen_cfg;
    gen_cfg.max_tokens = args.max_tokens;

    std::cout << "\033[1;32m"
              << "   ______           _     _  __\n"
              << "  / ____/__  ____  (_)__ | |/ /\n"
              << " / / __/ _ \\/ __ \\/ / _ \\|   / \n"
              << "/ /_/ /  __/ / / / /  __/   |  \n"
              << "\\____/\\___/_/ /_/_/\\___/_/|_| \n"
              << "\033[0m\n";

    // ── Load model ────────────────────────────────────────────────────────────

    std::cout << "\033[1;36mLoading Qwen2.5-VL-7B...\033[0m\n";
    auto model = geniex::qwen2_5_vl_7b::makeModel(runtime_cfg, config);
    if (!model) {
        std::cerr << "Failed to initialize model.\n";
        return 1;
    }
    std::cout << "\033[1;32mModel loaded.\033[0m\n\n";

    // ── Processor ────────────────────────────────────────────────────────────
    //
    // The vision graph is compiled for a fixed 24x36 patch grid, so we force
    // the preprocessor to always resize to 336x504 regardless of input aspect.
    geniex::qwen2vl::Qwen2VLConfig proc_cfg;
    proc_cfg.fixed_height = geniex::qwen2_5_vl_7b::kImageHeight;
    proc_cfg.fixed_width  = geniex::qwen2_5_vl_7b::kImageWidth;
    auto processor = geniex::qwen2vl::Qwen2VLProcessor::create(
        config.llm_config.tokenizer_path, proc_cfg);

    // ── Chat loop ─────────────────────────────────────────────────────────────

    bool first_turn = true;
    while (true) {
        std::cout << "Enter your prompt (type 'exit' to quit): ";
        std::string input;
        if (!std::getline(std::cin, input) || input == "exit" || input == "quit") break;

        std::string prompt_text;
        std::vector<std::string> image_paths;
        parseInput(input, prompt_text, image_paths);

        const bool saved_first_turn = first_turn;

        const auto t_start = std::chrono::high_resolution_clock::now();
        std::chrono::high_resolution_clock::time_point t_first_token;
        bool got_first_token = false;

        std::cout << "\033[33m";
        std::vector<int32_t> output_tokens;
        try {
            std::vector<int32_t> prompt_tokens;
            geniex::VLMInput vlm_input;

            if (first_turn) {
                geniex::ChatMessage system_msg{"system", args.system_prompt};
                geniex::ChatMessage user_msg{"user", prompt_text};
                for (const auto& img : image_paths)
                    user_msg.mm_content_paths.push_back(img);

                geniex::BatchFeatures bf = processor->process(
                    {{system_msg, user_msg}, /*add_generation_prompt=*/true});
                vlm_input.pixel_data = toPixelData(bf);
                prompt_tokens.assign(bf.input_ids.cbegin(), bf.input_ids.cend());
                first_turn = false;

            } else if (!image_paths.empty()) {
                geniex::ChatMessage user_msg{"user", prompt_text};
                for (const auto& img : image_paths)
                    user_msg.mm_content_paths.push_back(img);
                geniex::BatchFeatures bf = processor->process(
                    {{user_msg}, /*add_generation_prompt=*/true});
                vlm_input.pixel_data = toPixelData(bf);

                // Close the previous assistant turn before starting the new one.
                auto prefix = processor->tokenizer().encode("<|im_end|>\n",
                                                            /*add_special_tokens=*/false);
                prompt_tokens.assign(prefix.begin(), prefix.end());
                prompt_tokens.insert(prompt_tokens.end(),
                                     bf.input_ids.cbegin(), bf.input_ids.cend());

            } else {
                // Subsequent text-only turn: skip the processor entirely, the
                // KV cache already holds prior context.
                const std::string turn_text =
                    "<|im_end|>\n"
                    "<|im_start|>user\n" + prompt_text + "<|im_end|>\n"
                    "<|im_start|>assistant\n";
                prompt_tokens = processor->tokenizer().encode(
                    turn_text, /*add_special_tokens=*/false);
            }

            output_tokens = model->generate(
                prompt_tokens,
                vlm_input,
                gen_cfg,
                [&](int32_t tok) {
                    if (!got_first_token) {
                        t_first_token   = std::chrono::high_resolution_clock::now();
                        got_first_token = true;
                    }
                    std::cout << processor->tokenizer().decode({tok}) << std::flush;
                    return true;
                });
        } catch (const std::exception& e) {
            std::cout << "\033[0m\n" << std::flush;
            std::cerr << "Error: " << e.what() << "\n" << std::flush;
            model->resetKVCache();
            first_turn = saved_first_turn;
            continue;
        }
        std::cout << "\033[0m\n";

        const auto t_end = std::chrono::high_resolution_clock::now();

        if (got_first_token) {
            const double ttft_ms    = std::chrono::duration<double, std::milli>(t_first_token - t_start).count();
            const double decode_ms  = std::chrono::duration<double, std::milli>(t_end - t_first_token).count();
            const size_t decode_tok = output_tokens.size() > 1 ? output_tokens.size() - 1 : 0;
            const double tps        = decode_ms > 0.0 ? decode_tok / (decode_ms / 1000.0) : 0.0;

            if (args.verbose) {
                std::cout << "\033[1;36m=== Performance ===\033[0m\n"
                          << "Generated tokens : " << output_tokens.size() << "\n"
                          << "TTFT             : " << std::fixed << std::setprecision(1) << ttft_ms  << " ms\n"
                          << "Decode time      : " << std::fixed << std::setprecision(1) << decode_ms << " ms\n"
                          << "Decode speed     : " << std::fixed << std::setprecision(2) << tps << " tokens/s\n"
                          << "===================\n\n";
            } else {
                std::cout << "TTFT: " << std::fixed << std::setprecision(1) << ttft_ms << " ms"
                          << "  |  " << std::setprecision(2) << tps << " tokens/s\n\n";
            }
        }
    }

    return 0;
}
