// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// LLMPipeline-based variant of qwen3_4b_instruct_2507_example.cpp. Mirrors the
// GenieX CLI's `geniex infer` inference loop (cli/cmd/geniex/infer.go
// inferLLM()): chat-template-driven multi-turn REPL, sampler config sourced
// from the bundle's genie_config.json `dialog.sampler` block (with CLI flags
// able to override any individual field), and the same sliding-window /
// context-length-exceeded handling.
//
// Unlike the LLMModel-based example, this uses the high-level LLMPipeline API
// (tokenizer + chat template + streaming generation bundled together) instead
// of driving LLMModel::generate() directly with hand-tokenized prompts.

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "geniex-proc/types.h"
#include "llm/llm_spec_loader.h"  // parseGenieSamplerConfig
#include "qwen3/qwen3.h"
#include "types.h"

#ifdef _WIN32
#include <windows.h>
static void enable_utf8_io() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    DWORD  mode = 0;
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (GetConsoleMode(hOut, &mode)) SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}
#endif

struct Args {
    int32_t     max_tokens     = 512;
    bool        verbose        = false;
    bool        sliding_window = false;
    int32_t     n_keep         = 4;
    std::string prompt_file;
    std::string system_prompt = "You are a helpful AI assistant.";

    // Sampling. genie_config.json's `dialog.sampler` block is applied first
    // (see qairt::apply_sampler_config / parseGenieSamplerConfig); any flag
    // the user actually passes below overrides that bundle default field-by-
    // field. --no-sampling forces the greedy argmax fast path instead.
    bool     sampling               = true;
    bool     temperature_set        = false;
    float    temperature            = 0.0f;
    bool     top_p_set              = false;
    float    top_p                  = 0.0f;
    bool     top_k_set              = false;
    int32_t  top_k                  = 0;
    bool     min_p_set              = false;
    float    min_p                  = 0.0f;
    bool     repetition_penalty_set = false;
    float    repetition_penalty     = 1.0f;
    bool     presence_penalty_set   = false;
    float    presence_penalty       = 0.0f;
    bool     frequency_penalty_set  = false;
    float    frequency_penalty      = 0.0f;
    bool     penalty_last_n_set     = false;
    int32_t  penalty_last_n         = 64;
    bool     seed_set               = false;
    uint32_t seed                   = 0;
};

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --max-tokens <n>          Max tokens to generate (default 512)\n"
              << "  --verbose                 Print performance metrics\n"
              << "  --sliding-window          Evict oldest context on overflow instead of erroring\n"
              << "  --n-keep <n>              Tokens to keep anchored when sliding (default 4)\n"
              << "  --prompt-file <path>      Read the first turn's prompt from a file (prints its token\n"
              << "                            count, then falls through to the interactive REPL)\n"
              << "  --system-prompt <s>       System prompt (default: \"You are a helpful AI assistant.\")\n"
              << "\n"
              << "  Sampling defaults come from the bundle's genie_config.json (dialog.sampler); any of\n"
              << "  the flags below overrides that one field. --no-sampling forces greedy argmax.\n"
              << "  --no-sampling             Disable sampling entirely (greedy argmax)\n"
              << "  --temperature <f>         Sampling temperature\n"
              << "  --top-p <f>               Top-p / nucleus cutoff\n"
              << "  --top-k <n>               Top-k cutoff\n"
              << "  --min-p <f>               Min-p cutoff\n"
              << "  --repetition-penalty <f>  Repetition penalty\n"
              << "  --presence-penalty <f>    Presence penalty\n"
              << "  --frequency-penalty <f>   Frequency penalty\n"
              << "  --penalty-last-n <n>      Token window the penalties look back over\n"
              << "  --seed <n>                RNG seed for sampling\n"
              << "  --help\n";
}

static bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a    = argv[i];
        auto        next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string{}; };
        if (a == "--max-tokens")
            args.max_tokens = std::stoi(next());
        else if (a == "--verbose")
            args.verbose = true;
        else if (a == "--sliding-window")
            args.sliding_window = true;
        else if (a == "--n-keep")
            args.n_keep = std::stoi(next());
        else if (a == "--prompt-file")
            args.prompt_file = next();
        else if (a == "--system-prompt")
            args.system_prompt = next();
        else if (a == "--no-sampling")
            args.sampling = false;
        else if (a == "--temperature") {
            args.temperature     = std::stof(next());
            args.temperature_set = true;
        } else if (a == "--top-p") {
            args.top_p     = std::stof(next());
            args.top_p_set = true;
        } else if (a == "--top-k") {
            args.top_k     = std::stoi(next());
            args.top_k_set = true;
        } else if (a == "--min-p") {
            args.min_p     = std::stof(next());
            args.min_p_set = true;
        } else if (a == "--repetition-penalty") {
            args.repetition_penalty     = std::stof(next());
            args.repetition_penalty_set = true;
        } else if (a == "--presence-penalty") {
            args.presence_penalty     = std::stof(next());
            args.presence_penalty_set = true;
        } else if (a == "--frequency-penalty") {
            args.frequency_penalty     = std::stof(next());
            args.frequency_penalty_set = true;
        } else if (a == "--penalty-last-n") {
            args.penalty_last_n     = std::stoi(next());
            args.penalty_last_n_set = true;
        } else if (a == "--seed") {
            args.seed     = static_cast<uint32_t>(std::stoul(next()));
            args.seed_set = true;
        } else if (a == "--help" || a == "-h") {
            printUsage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            return false;
        }
    }
    return true;
}

// Reads an entire file into a string. Throws std::runtime_error if it cannot be opened.
static std::string readFileToString(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open prompt file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char** argv) {
#ifdef _WIN32
    enable_utf8_io();
#endif

    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    const auto model_dir =
        std::filesystem::path("C:\\Users\\yichqian\\.cache\\geniex\\models\\qualcomm\\Qwen3-4B-Instruct-2507");

    // All QNN runtime paths are left as std::nullopt → auto-detected from
    // htp-files/ installed alongside geniex_core.
    geniex::QnnRuntimeConfig runtime_cfg;

    geniex::ModelConfig model_cfg;
    model_cfg.model_paths = {
        (model_dir / "part1_of_4.bin").string(),
        (model_dir / "part2_of_4.bin").string(),
        (model_dir / "part3_of_4.bin").string(),
        (model_dir / "part4_of_4.bin").string(),
    };
    model_cfg.tokenizer_path = (model_dir / "tokenizer.json").string();
    // No embedding_path needed – embedding runs on-device in shard 0.
    model_cfg.htp_config_path = (model_dir / "htp_backend_ext_config.json").string();

    // Sampler defaults from the bundle's genie_config.json `dialog.sampler` block
    // (the same source cli/cmd/geniex reads indirectly via the qairt plugin's
    // apply_sampler_config()). Returns an all-nullopt struct if the bundle has no
    // sampler block, in which case the plugin-level defaults below apply.
    const geniex::ParsedSamplerConfig bundle_sampler = geniex::parseGenieSamplerConfig(model_dir);

    // Same fallback constants as sdk/plugins/qairt/include/sampler_config_utils.h's
    // DefaultSamplerParams, used only when the bundle *and* the CLI flags are both
    // silent on a given field.
    constexpr uint32_t kDefaultSeed              = 42;
    constexpr int32_t  kDefaultTopK              = 40;
    constexpr float    kDefaultTopP              = 0.95f;
    constexpr float    kDefaultMinP              = 0.0f;
    constexpr float    kDefaultTemperature       = 0.8f;
    constexpr float    kDefaultRepetitionPenalty = 1.0f;
    constexpr float    kDefaultPresencePenalty   = 0.0f;
    constexpr float    kDefaultFrequencyPenalty  = 0.0f;

    geniex::GenerationConfig gen_cfg;
    gen_cfg.max_tokens            = args.max_tokens;
    gen_cfg.sliding_window        = args.sliding_window;
    gen_cfg.sliding_window_n_keep = args.n_keep;

    gen_cfg.enable_sampling = args.sampling;
    if (args.sampling) {
        gen_cfg.seed  = args.seed_set ? args.seed : bundle_sampler.seed.value_or(kDefaultSeed);
        gen_cfg.top_k = args.top_k_set ? args.top_k : bundle_sampler.top_k.value_or(kDefaultTopK);
        gen_cfg.top_p = args.top_p_set ? args.top_p : bundle_sampler.top_p.value_or(kDefaultTopP);
        gen_cfg.min_p = args.min_p_set ? args.min_p : kDefaultMinP;  // genie has no min_p field
        gen_cfg.temperature =
            args.temperature_set ? args.temperature : bundle_sampler.temperature.value_or(kDefaultTemperature);
        gen_cfg.repetition_penalty = args.repetition_penalty_set
                                         ? args.repetition_penalty
                                         : bundle_sampler.repetition_penalty.value_or(kDefaultRepetitionPenalty);
        gen_cfg.presence_penalty   = args.presence_penalty_set
                                         ? args.presence_penalty
                                         : bundle_sampler.presence_penalty.value_or(kDefaultPresencePenalty);
        gen_cfg.frequency_penalty  = args.frequency_penalty_set
                                         ? args.frequency_penalty
                                         : bundle_sampler.frequency_penalty.value_or(kDefaultFrequencyPenalty);
        if (args.penalty_last_n_set)
            gen_cfg.penalty_last_n = args.penalty_last_n;
        else if (bundle_sampler.penalty_last_n)
            gen_cfg.penalty_last_n = *bundle_sampler.penalty_last_n;
    }

    if (args.sliding_window) {
        std::cout << "\033[1;35mSliding window enabled (n_keep=" << args.n_keep << ")\033[0m\n";
    }
    if (args.sampling) {
        std::cout << "\033[1;35mSampling enabled (temperature=" << gen_cfg.temperature << " top_p=" << gen_cfg.top_p
                  << " top_k=" << gen_cfg.top_k << " min_p=" << gen_cfg.min_p
                  << " repetition_penalty=" << gen_cfg.repetition_penalty
                  << " presence_penalty=" << gen_cfg.presence_penalty
                  << " frequency_penalty=" << gen_cfg.frequency_penalty << " penalty_last_n=" << gen_cfg.penalty_last_n
                  << " seed=" << gen_cfg.seed << ")\033[0m\n";
    } else {
        std::cout << "\033[1;35mSampling disabled (greedy argmax)\033[0m\n";
    }

    std::cout << "\033[1;32m"
              << "   ______           _     _  __\n"
              << "  / ____/__  ____  (_)__ | |/ /\n"
              << " / / __/ _ \\/ __ \\/ / _ \\|   / \n"
              << "/ /_/ /  __/ / / / /  __/   |  \n"
              << "\\____/\\___/_/ /_/_/\\___/_/|_| \n"
              << "\033[0m\n";

    std::cout << "\033[1;36mLoading model...\033[0m\n";
    auto maybe_pipe = geniex::qwen3::makePipeline(runtime_cfg, model_cfg);
    if (!maybe_pipe) {
        std::cerr << "Failed to initialize pipeline.\n";
        return 1;
    }
    auto& pipe = *maybe_pipe;
    std::cout << "\033[1;32mModel loaded.\033[0m\n\n";

    // `history` accumulates every turn, but only the messages since the last assistant turn are
    // (re-)rendered and fed to generate() each round -- LLMModel is stateful (its KV cache already
    // holds every prior turn's tokens), so re-feeding the full history would re-prefill already-
    // cached tokens on top of themselves, inflating prompt_tokens and defeating sliding-window
    // eviction. Mirrors QairtLlm::apply_chat_template's start_idx trimming (sdk/plugins/qairt).
    std::vector<geniex::ChatMessage> history;
    if (!args.system_prompt.empty()) {
        history.push_back({geniex::Role::System, args.system_prompt});
    }
    bool is_first_turn = true;

    // Optional --prompt-file: run one turn from the file's raw content first, printing its token
    // count, then fall through to the normal interactive REPL below for follow-up turns.
    bool first_turn_pending = !args.prompt_file.empty();

    while (true) {
        std::string input;
        if (first_turn_pending) {
            first_turn_pending = false;
            try {
                input = readFileToString(args.prompt_file);
            } catch (const std::exception& e) {
                std::cerr << e.what() << "\n";
                return 1;
            }
            std::cout << "Enter your prompt (type 'exit' to quit, 'reset' to clear): [from --prompt-file "
                      << args.prompt_file << ", " << input.size() << " bytes]\n";
        } else {
            std::cout << "Enter your prompt (type 'exit' to quit, 'reset' to clear): ";
            if (!std::getline(std::cin, input) || input == "exit" || input == "quit") break;
        }
        if (input == "reset") {
            pipe.reset();
            history.clear();
            if (!args.system_prompt.empty()) history.push_back({geniex::Role::System, args.system_prompt});
            is_first_turn = true;
            std::cout << "\033[1;36m[conversation reset]\033[0m\n\n";
            continue;
        }

        history.push_back({geniex::Role::User, input});

        // Render only the messages since the last assistant turn (see comment above `history`).
        size_t start_idx = 0;
        if (!is_first_turn) {
            for (size_t i = history.size(); i-- > 0;) {
                if (history[i].role == geniex::Role::Assistant) {
                    start_idx = i + 1;
                    break;
                }
            }
        }
        const std::vector<geniex::ChatMessage> turn_messages(
            history.begin() + static_cast<std::ptrdiff_t>(start_idx), history.end());

        geniex::ApplyChatTemplateOptions tmpl_opts;
        tmpl_opts.add_generation_prompt = true;
        std::string formatted;
        try {
            formatted = pipe.applyChatTemplate(turn_messages, tmpl_opts);
        } catch (const std::exception& e) {
            std::cerr << "applyChatTemplate failed: " << e.what() << "\n";
            history.pop_back();
            continue;
        }
        is_first_turn = false;

        std::cout << "\033[33m";
        auto on_token = [](const char* piece) -> bool {
            std::cout << piece << std::flush;
            return true;
        };

        geniex::GenerateResult result = pipe.generate(formatted, gen_cfg, on_token);
        std::cout << "\033[0m\n";
        std::cout << "\033[2m[prompt_tokens=" << result.prompt_tokens << " n_past=" << pipe.nPast() << "]\033[0m\n";

        if (result.stop_reason == "error") {
            std::cerr << "Generation error — resetting conversation.\n";
            pipe.reset();
            history.clear();
            if (!args.system_prompt.empty()) history.push_back({geniex::Role::System, args.system_prompt});
            is_first_turn = true;
            continue;
        }
        if (result.stop_reason == "context_length") {
            std::cerr << "Generation error: context length exceeded — resetting conversation.\n";
            pipe.reset();
            history.clear();
            if (!args.system_prompt.empty()) history.push_back({geniex::Role::System, args.system_prompt});
            is_first_turn = true;
            continue;
        }

        history.push_back({geniex::Role::Assistant, result.full_text});

        if (args.verbose) {
            std::cout << "\033[1;36m=== Performance ===\033[0m\n"
                      << "Generated tokens : " << result.generated_tokens << "\n"
                      << "TTFT             : " << std::fixed << std::setprecision(1) << result.ttft_ms << " ms\n"
                      << "Decode time      : " << std::fixed << std::setprecision(1) << result.decode_ms << " ms\n"
                      << "Decode speed     : " << std::fixed << std::setprecision(2) << result.tokens_per_second
                      << " tokens/s\n"
                      << "Stop reason      : " << result.stop_reason << "\n"
                      << "===================\n\n";
        } else {
            std::cout << "TTFT: " << std::fixed << std::setprecision(1) << result.ttft_ms << " ms"
                      << "  |  " << std::setprecision(2) << result.tokens_per_second << " tokens/s\n\n";
        }
    }

    return 0;
}
