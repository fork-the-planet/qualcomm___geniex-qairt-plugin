// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>

#include "pipeline/vlm_pipeline.h"
#include "types.h"

// Per-model makePipeline factories.
#include "qwen2_5_vl/qwen2_5_vl.h"

namespace geniex {

struct VlmModelEntry {
    std::function<std::optional<VLMPipeline>(const QnnRuntimeConfig&, const VLMConfig&)> make_pipeline;
};

inline const std::unordered_map<std::string, VlmModelEntry>& vlm_model_registry() {
    static const std::unordered_map<std::string, VlmModelEntry> registry = {
        // model names here should match QAIHub model IDs
        {"qwen2_5_vl_7b_instruct", {qwen2_5_vl_7b::makePipeline}},
    };
    return registry;
}

}  // namespace geniex
