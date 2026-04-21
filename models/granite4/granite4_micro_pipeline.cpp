#include <atomic>
#include <csignal>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

#include "pipeline/llm_pipeline.h"
#include "granite4.h"

#ifdef _WIN32
#include <windows.h>
static void enable_utf8_io() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    DWORD mode = 0;
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (GetConsoleMode(hOut, &mode))
        SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}
#endif

// Set by SIGINT (CTRL-C). Checked by the streaming callback to stop generation.
static std::atomic<bool> g_stop_requested{false};
static void sigint_handler(int sig) {
    g_stop_requested.store(true);
    // On Windows the signal disposition is reset to SIG_DFL after the handler
    // fires; without re-arming, a second CTRL-C hits the default handler and
    // terminates the process mid-inference, leaving QNN graphs un-destroyed.
    std::signal(sig, sigint_handler);
}

struct Args {
    int32_t     max_tokens     = 512;
    bool        verbose        = false;
    std::string system_prompt  = "You are Granite, developed by IBM.";
};

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --max-tokens <n>      Max tokens to generate (default 512)\n"
              << "  --system-prompt <s>   System prompt\n"
              << "  --verbose             Print performance metrics\n"
              << "  --help\n";
}

static bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : std::string{};
        };
        if      (a == "--max-tokens")     args.max_tokens     = std::stoi(next());
        else if (a == "--system-prompt")  args.system_prompt  = next();
        else if (a == "--verbose")        args.verbose        = true;
        else if (a == "--help" || a == "-h") { printUsage(argv[0]); return false; }
        else { std::cerr << "Unknown argument: " << a << "\n"; return false; }
    }
    return true;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    enable_utf8_io();
#endif

    std::signal(SIGINT, sigint_handler);

    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    const auto model_dir = std::filesystem::current_path() / "modelfiles" / "granite4_micro";

    // All QNN runtime paths are left as std::nullopt → auto-detected from
    // htp-files/ installed alongside geniex_core.
    geniex::QnnRuntimeConfig runtime_cfg;

    geniex::ModelConfig model_cfg;
    model_cfg.model_paths = {
        (model_dir / "weight_sharing_model_1_of_2.serialized.bin").string(),
        (model_dir / "weight_sharing_model_2_of_2.serialized.bin").string(),
    };
    model_cfg.tokenizer_path   = (model_dir / "tokenizer.json").string();
    model_cfg.embedding_path   = (model_dir / "embed_tokens.npy").string();
    model_cfg.htp_config_path  = (model_dir / "htp_backend_ext_config.json").string();

    geniex::GenerationConfig gen_cfg;
    gen_cfg.max_tokens = args.max_tokens;

    std::cout << "\033[1;32m"
              << "   ______           _     _  __\n"
              << "  / ____/__  ____  (_)__ | |/ /\n"
              << " / / __/ _ \\/ __ \\/ / _ \\|   / \n"
              << "/ /_/ /  __/ / / / /  __/   |  \n"
              << "\\____/\\___/_/ /_/_/\\___/_/|_| \n"
              << "\033[0m\n";

    // ── Create pipeline ──────────────────────────────────────────────────────
    std::cout << "\033[1;36mLoading model via LLMPipeline...\033[0m\n";

    auto pipe = geniex::granite4_micro::makePipeline(runtime_cfg, model_cfg);
    if (!pipe) {
        std::cerr << "Failed to create pipeline.\n";
        return 1;
    }

    pipe->setSystemPrompt(args.system_prompt);

    std::cout << "\033[1;32mPipeline ready.\033[0m\n\n";

    // ── Chat loop ────────────────────────────────────────────────────────────
    while (true) {
        std::cout << "Enter your prompt (type 'exit' to quit): ";
        std::string input;
        if (!std::getline(std::cin, input) || input == "exit" || input == "quit")
            break;

        std::string prompt = pipe->applyChatTemplate(input);

        std::cout << "\033[33m";
        g_stop_requested.store(false);
        auto result = pipe->generate(prompt, gen_cfg,
            [](const char* piece) {
                std::cout << piece << std::flush;
                // CTRL-C → SIGINT → g_stop_requested = true → stop streaming.
                return !g_stop_requested.load();
            });
        std::cout << "\033[0m\n";

        // ── Print metrics ────────────────────────────────────────────────────
        if (args.verbose) {
            std::cout << "\033[1;36m=== Performance ===\033[0m\n"
                      << "Stop reason      : " << result.stop_reason << "\n"
                      << "Prompt tokens    : " << result.prompt_tokens << "\n"
                      << "Generated tokens : " << result.generated_tokens << "\n"
                      << "TTFT             : " << std::fixed << std::setprecision(1) << result.ttft_ms  << " ms\n"
                      << "Decode time      : " << std::fixed << std::setprecision(1) << result.decode_ms << " ms\n"
                      << "Decode speed     : " << std::fixed << std::setprecision(2) << result.tokens_per_second << " tokens/s\n"
                      << "KV n_past        : " << pipe->nPast() << "\n"
                      << "===================\n\n";
        } else {
            std::cout << "TTFT: " << std::fixed << std::setprecision(1) << result.ttft_ms << " ms"
                      << "  |  " << std::setprecision(2) << result.tokens_per_second << " tokens/s"
                      << "  |  " << result.stop_reason << "\n\n";
        }

        // Test reset on "reset" input
        if (input == "reset") {
            pipe->reset();
            std::cout << "\033[1;33mPipeline reset (KV cleared, first_turn restored).\033[0m\n\n";
        }
    }

    return 0;
}
