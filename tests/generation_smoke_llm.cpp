// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Generation smoke test for every LLM registered in llm_model_registry.h.
//
// Invocation (via CTest):
//     generation_smoke_llm --model <name> --prompt "<text>" --min-tokens <n>
//
// Returns:
//     0  success (generated_tokens >= min_tokens)
//     1  init / runtime error (model files missing, QNN init failed, etc.)
//     2  smoke-test assertion failure (too few tokens generated)
//
// The test resolves model files from `./modelfiles/<name>/` relative to the
// current working directory — same layout every example uses. The CTest
// wiring sets WORKING_DIRECTORY to the repo root.
//
// Models are added centrally in `models/llm_model_registry.h`. The path
// table below mirrors those entries and must stay in sync. Keeping them in
// one place here (rather than spreading across all examples) is intentional:
// any contributor adding a model only needs to edit llm_model_registry.h +
// this table to get CTest coverage.

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "llm_model_registry.h"
#include "pipeline/llm_pipeline.h"
#include "types.h"

namespace fs = std::filesystem;

// Per-model file layout. Centralised so contributors adding a new model only
// edit this table once (plus llm_model_registry.h). Shards are listed in
// execution order.
struct ModelFiles {
    std::string                modelfiles_subdir;  // under ./modelfiles/
    std::vector<std::string>   bin_shards;         // relative to subdir
    std::string                tokenizer  = "tokenizer.json";
    std::string                htp_config = "htp_backend_ext_config.json";
    std::string                embedding;          // empty = on-device embedding
};

static const std::vector<std::pair<std::string, ModelFiles>>& modelFilesTable() {
    // clang-format off
    static const std::vector<std::pair<std::string, ModelFiles>> table = {
        {"qwen3_4b", {
            "qwen3_4b",
            {"qwen3_4b_part_1_of_4.bin", "qwen3_4b_part_2_of_4.bin",
             "qwen3_4b_part_3_of_4.bin", "qwen3_4b_part_4_of_4.bin"}}},

        {"qwen3_4b_instruct_2507", {
            "qwen3_4b_instruct_2507",
            {"qwen3_4b_instruct_2507_part_1_of_4.bin",
             "qwen3_4b_instruct_2507_part_2_of_4.bin",
             "qwen3_4b_instruct_2507_part_3_of_4.bin",
             "qwen3_4b_instruct_2507_part_4_of_4.bin"}}},

        {"qwen2_5_7b_instruct", {
            "qwen2_5_7b_instruct",
            {"qwen2_5_7b_instruct_part_1_of_6.bin",
             "qwen2_5_7b_instruct_part_2_of_6.bin",
             "qwen2_5_7b_instruct_part_3_of_6.bin",
             "qwen2_5_7b_instruct_part_4_of_6.bin",
             "qwen2_5_7b_instruct_part_5_of_6.bin",
             "qwen2_5_7b_instruct_part_6_of_6.bin"}}},

        {"phi_3_5_mini_instruct", {
            "phi3_5",
            {"weight_sharing_model_1_of_4.serialized.bin",
             "weight_sharing_model_2_of_4.serialized.bin",
             "weight_sharing_model_3_of_4.serialized.bin",
             "weight_sharing_model_4_of_4.serialized.bin"}}},

        {"llama_v3_8b_instruct", {
            "llama_v3_8b_instruct",
            {"llama_v3_8b_instruct_part_1_of_5.bin",
             "llama_v3_8b_instruct_part_2_of_5.bin",
             "llama_v3_8b_instruct_part_3_of_5.bin",
             "llama_v3_8b_instruct_part_4_of_5.bin",
             "llama_v3_8b_instruct_part_5_of_5.bin"}}},

        {"llama_v3_elyza_jp_8b", {
            "llama_v3_elyza_jp_8b",
            {"llama_v3_elyza_jp_8b_part_1_of_5.bin",
             "llama_v3_elyza_jp_8b_part_2_of_5.bin",
             "llama_v3_elyza_jp_8b_part_3_of_5.bin",
             "llama_v3_elyza_jp_8b_part_4_of_5.bin",
             "llama_v3_elyza_jp_8b_part_5_of_5.bin"}}},

        {"llama_v3_taide_8b_chat", {
            "llama_v3_taide_8b_chat",
            {"llama_v3_taide_8b_chat_part_1_of_5.bin",
             "llama_v3_taide_8b_chat_part_2_of_5.bin",
             "llama_v3_taide_8b_chat_part_3_of_5.bin",
             "llama_v3_taide_8b_chat_part_4_of_5.bin",
             "llama_v3_taide_8b_chat_part_5_of_5.bin"}}},

        {"llama_v3_1_8b_instruct", {
            "llama_v3_1_8b_instruct-genie-w4a16-qualcomm_snapdragon_x_elite",
            {"llama_v3_1_8b_instruct_part_1_of_5.bin",
             "llama_v3_1_8b_instruct_part_2_of_5.bin",
             "llama_v3_1_8b_instruct_part_3_of_5.bin",
             "llama_v3_1_8b_instruct_part_4_of_5.bin",
             "llama_v3_1_8b_instruct_part_5_of_5.bin"}}},

        {"llama_v3_1_sea_lion_3_5_8b_r", {
            "llama_v3_1_sea_lion_3_5_8b_r-genie-w4a16-qualcomm_snapdragon_x_elite",
            {"llama_v3_1_sea_lion_3_5_8b_r_part_1_of_5.bin",
             "llama_v3_1_sea_lion_3_5_8b_r_part_2_of_5.bin",
             "llama_v3_1_sea_lion_3_5_8b_r_part_3_of_5.bin",
             "llama_v3_1_sea_lion_3_5_8b_r_part_4_of_5.bin",
             "llama_v3_1_sea_lion_3_5_8b_r_part_5_of_5.bin"}}},

        {"llama_v3_2_1b_instruct", {
            "llama_v3_2_1b_instruct-genie-w4-qualcomm_snapdragon_x_elite",
            {"llama_v3_2_1b_instruct_part_1_of_3.bin",
             "llama_v3_2_1b_instruct_part_2_of_3.bin",
             "llama_v3_2_1b_instruct_part_3_of_3.bin"}}},

        {"llama_v3_2_3b_instruct", {
            "llama_v3_2_3b_instruct_exported_cb",
            {"llama_v3_2_3b_instruct_part_1_of_3.bin",
             "llama_v3_2_3b_instruct_part_2_of_3.bin",
             "llama_v3_2_3b_instruct_part_3_of_3.bin"}}},

        {"falcon_v3_7b_instruct", {
            "falcon_v3_7b_instruct-genie-w4a16-qualcomm_snapdragon_x_elite",
            {"falcon_v3_7b_instruct_part_1_of_5.bin",
             "falcon_v3_7b_instruct_part_2_of_5.bin",
             "falcon_v3_7b_instruct_part_3_of_5.bin",
             "falcon_v3_7b_instruct_part_4_of_5.bin",
             "falcon_v3_7b_instruct_part_5_of_5.bin"}}},
    };
    // clang-format on
    return table;
}

static const ModelFiles* findModelFiles(const std::string& name) {
    for (const auto& [key, files] : modelFilesTable())
        if (key == name) return &files;
    return nullptr;
}

struct Args {
    std::string model;
    std::string prompt      = "Hello, briefly introduce yourself.";
    int32_t     max_tokens  = 32;
    int32_t     min_tokens  = 5;
    bool        list_models = false;
};

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " --model <name> [OPTIONS]\n"
              << "  --model <name>      Model id (see --list-models)\n"
              << "  --prompt <text>     Prompt to generate on (default: brief intro)\n"
              << "  --max-tokens <n>    Max tokens to generate (default 32)\n"
              << "  --min-tokens <n>    Fail with exit 2 if fewer tokens generated (default 5)\n"
              << "  --list-models       Print the list of known model ids and exit\n"
              << "  --help\n";
}

static bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : std::string{};
        };
        if      (a == "--model")       args.model       = next();
        else if (a == "--prompt")      args.prompt      = next();
        else if (a == "--max-tokens")  args.max_tokens  = std::stoi(next());
        else if (a == "--min-tokens")  args.min_tokens  = std::stoi(next());
        else if (a == "--list-models") args.list_models = true;
        else if (a == "--help" || a == "-h") { printUsage(argv[0]); return false; }
        else { std::cerr << "Unknown argument: " << a << "\n"; return false; }
    }
    return true;
}

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    if (args.list_models) {
        for (const auto& [name, _] : geniex::llm_model_registry())
            std::cout << name << "\n";
        return 0;
    }

    if (args.model.empty()) {
        std::cerr << "--model is required. Use --list-models to see options.\n";
        return 1;
    }

    // Look up factory (registry) and file layout (local table).
    const auto& registry = geniex::llm_model_registry();
    auto reg_it = registry.find(args.model);
    if (reg_it == registry.end()) {
        std::cerr << "Unknown model '" << args.model << "'. Use --list-models.\n";
        return 1;
    }
    const ModelFiles* files = findModelFiles(args.model);
    if (!files) {
        std::cerr << "Model '" << args.model
                  << "' is registered but has no file-path entry in "
                     "generation_smoke_llm.cpp. Add it to modelFilesTable().\n";
        return 1;
    }

    const fs::path model_dir =
        fs::current_path() / "modelfiles" / files->modelfiles_subdir;

    geniex::QnnRuntimeConfig runtime_cfg;  // auto-detect HTP paths

    geniex::ModelConfig model_cfg;
    for (const auto& shard : files->bin_shards)
        model_cfg.model_paths.push_back((model_dir / shard).string());
    model_cfg.tokenizer_path  = (model_dir / files->tokenizer).string();
    model_cfg.htp_config_path = (model_dir / files->htp_config).string();
    if (!files->embedding.empty())
        model_cfg.embedding_path = (model_dir / files->embedding).string();

    // Fail fast with a clear message if any required file is missing, so the
    // failure mode is "files not staged" rather than a deep QNN init error.
    for (const auto& p : model_cfg.model_paths) {
        if (!fs::exists(p)) {
            std::cerr << "Missing shard: " << p << "\n"
                      << "Stage model files under ./modelfiles/"
                      << files->modelfiles_subdir << "/ before running this test.\n";
            return 1;
        }
    }
    if (!fs::exists(model_cfg.tokenizer_path)) {
        std::cerr << "Missing tokenizer: " << model_cfg.tokenizer_path << "\n";
        return 1;
    }

    std::cout << "[smoke] model=" << args.model
              << "  prompt=\"" << args.prompt << "\""
              << "  max_tokens=" << args.max_tokens
              << "  min_tokens=" << args.min_tokens << "\n";

    std::optional<geniex::LLMPipeline> pipe;
    try {
        pipe = reg_it->second.make_pipeline(runtime_cfg, model_cfg);
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
        result = pipe->generate(args.prompt, gen_cfg, /*on_token=*/nullptr);
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
