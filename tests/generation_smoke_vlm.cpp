// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Generation smoke test for VLM models (currently: Qwen2.5-VL-7B).
//
// Invocation (via CTest):
//     generation_smoke_vlm --model <name> --prompt "<text>" --image <path> --min-tokens <n>
//
// Returns:
//     0  success (generated_tokens >= min_tokens)
//     1  init / runtime error (model files missing, image missing, etc.)
//     2  smoke-test assertion failure (too few tokens generated)

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "pipeline/vlm_pipeline.h"
#include "qwen2_5_vl/qwen2_5_vl.h"
#include "types.h"

namespace fs = std::filesystem;

struct Args {
    std::string model      = "qwen2_5_vl_7b";
    std::string prompt     = "What is in this image?";
    std::string image_path;
    int32_t     max_tokens = 32;
    int32_t     min_tokens = 5;
};

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " --image <path> [OPTIONS]\n"
              << "  --model <name>      VLM model id (default qwen2_5_vl_7b)\n"
              << "  --prompt <text>     Prompt to generate on\n"
              << "  --image  <path>     Image file to include with prompt (REQUIRED)\n"
              << "  --max-tokens <n>    Max tokens to generate (default 32)\n"
              << "  --min-tokens <n>    Fail with exit 2 if fewer tokens generated (default 5)\n"
              << "  --help\n";
}

static bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : std::string{};
        };
        if      (a == "--model")      args.model      = next();
        else if (a == "--prompt")     args.prompt     = next();
        else if (a == "--image")      args.image_path = next();
        else if (a == "--max-tokens") args.max_tokens = std::stoi(next());
        else if (a == "--min-tokens") args.min_tokens = std::stoi(next());
        else if (a == "--help" || a == "-h") { printUsage(argv[0]); return false; }
        else { std::cerr << "Unknown argument: " << a << "\n"; return false; }
    }
    return true;
}

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    if (args.image_path.empty()) {
        std::cerr << "--image is required for VLM smoke test.\n";
        return 1;
    }
    if (!fs::exists(args.image_path)) {
        std::cerr << "Image not found: " << args.image_path << "\n";
        return 1;
    }

    // Dispatch by model id. Only qwen2_5_vl_7b is wired up today; new VLM
    // factories can be added here alongside new entries in llm_model_registry.h
    // (or a future vlm_model_registry.h).
    if (args.model != "qwen2_5_vl_7b") {
        std::cerr << "Unknown VLM model '" << args.model
                  << "' (only 'qwen2_5_vl_7b' is supported).\n";
        return 1;
    }

    const fs::path model_dir =
        fs::current_path() / "modelfiles" / "qwen2_5_vl_7b";

    geniex::QnnRuntimeConfig runtime_cfg;  // auto-detect HTP paths

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

    config.vision_config.model_paths     = {(model_dir / "vision_encoder.bin").string()};
    config.vision_config.htp_config_path = (model_dir / "htp_backend_ext_config.json").string();

    // Fail fast with a clear message if any required file is missing.
    for (const auto& p : config.llm_config.model_paths) {
        if (!fs::exists(p)) {
            std::cerr << "Missing shard: " << p << "\n";
            return 1;
        }
    }
    for (const auto& p : {config.llm_config.tokenizer_path,
                          config.llm_config.htp_config_path,
                          config.llm_config.embedding_path,
                          config.vision_config.model_paths.front()}) {
        if (!fs::exists(p)) {
            std::cerr << "Missing VLM asset: " << p << "\n";
            return 1;
        }
    }

    std::cout << "[smoke] model=" << args.model
              << "  prompt=\"" << args.prompt << "\""
              << "  image=" << args.image_path
              << "  max_tokens=" << args.max_tokens
              << "  min_tokens=" << args.min_tokens << "\n";

    std::optional<geniex::VLMPipeline> pipe;
    try {
        pipe = geniex::qwen2_5_vl_7b::makePipeline(runtime_cfg, config);
    } catch (const std::exception& e) {
        std::cerr << "Pipeline construction threw: " << e.what() << "\n";
        return 1;
    }
    if (!pipe) {
        std::cerr << "Pipeline construction returned nullopt.\n";
        return 1;
    }

    geniex::GenerationConfig gen_cfg;
    gen_cfg.max_tokens = args.max_tokens;

    geniex::GenerateResult result;
    try {
        result = pipe->generate(args.prompt,
                                std::vector<std::string>{args.image_path},
                                gen_cfg,
                                /*on_token=*/nullptr);
    } catch (const std::exception& e) {
        std::cerr << "generate() threw: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[smoke] generated_tokens=" << result.generated_tokens
              << "  stop_reason=" << result.stop_reason
              << "  tps=" << result.tokens_per_second << "\n";

    if (static_cast<int32_t>(result.generated_tokens) < args.min_tokens) {
        std::cerr << "Smoke test FAILED: generated " << result.generated_tokens
                  << " tokens, expected at least " << args.min_tokens << "\n";
        return 2;
    }
    return 0;
}
