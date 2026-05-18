// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>

#include "pipeline/llm_pipeline.h"
#include "types.h"

// Per-model makePipeline factories.
#include "falcon3/falcon3.h"
#include "llama3/llama3.h"
#include "llama3_1/llama3_1.h"
#include "llama3_2/llama3_2.h"
#include "llama3_2_ssd/llama3_2_ssd.h"
#include "phi3_5/phi3_5.h"
#include "qwen2_5/qwen2_5.h"
#include "qwen3/qwen3.h"

namespace geniex {

struct LlmModelEntry {
    std::function<std::optional<LLMPipeline>(const QnnRuntimeConfig&, const ModelConfig&)> make_pipeline;
};

inline const std::unordered_map<std::string, LlmModelEntry>& llm_model_registry() {
    static const std::unordered_map<std::string, LlmModelEntry> registry = {
        // model names here should match QAIHub model IDs
        {"qwen3_4b", {qwen3_4b::makePipeline}},
        {"qwen3_4b_instruct_2507", {qwen3_4b_instruct_2507::makePipeline}},
        {"qwen3_8b", {qwen3_8b::makePipeline}},
        {"qwen2_5_7b_instruct", {qwen2_5_7b_instruct::makePipeline}},
        {"phi_3_5_mini_instruct", {phi3_5::makePipeline}},
        {"llama_v3_8b_instruct", {llama_v3_8b_instruct::makePipeline}},
        {"llama_v3_elyza_jp_8b", {llama_v3_elyza_jp_8b::makePipeline}},
        {"llama_v3_taide_8b_chat", {llama_v3_taide_8b_chat::makePipeline}},
        {"llama_v3_1_8b_instruct", {llama3_1_8b::makePipeline}},
        {"llama_v3_1_sea_lion_3_5_8b_r", {llama3_1_8b::makePipeline}},  // same arch as llama3_1_8b
        {"llama_v3_2_1b_instruct",    {llama3_2_1b::makePipeline}},
        {"llama_v3_2_3b_instruct",    {llama3_2_3b::makePipeline}},
        {"llama_v3_2_3b_instruct_ssd", {llama3_2_3b_ssd::makePipeline}},
        {"falcon_v3_7b_instruct",       {falcon3_7b::makePipeline}},
    };
    return registry;
}

}  // namespace geniex
