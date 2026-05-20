// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "llm/input_provider.h"
#include "llm/llm_model.h"
#include "llm/llm_spec_loader.h"
#include "llm/llm_types.h"
#include "logging.h"
#include "pipeline/chat_template.h"
#include "pipeline/llm_pipeline.h"

namespace geniex {
namespace qwen3 {

// Builds an LLMModel from a bundle pointed to by model_cfg.model_paths.
// All architectural details come from config.json + metadata.json — only
// the input-provider lineup is family-fixed here.
inline LLMModel makeModel(const ModelConfig& model_cfg) {
    const auto bundle = bundleDirOf(model_cfg);
    auto       hf     = parseHFConfig(bundle);
    auto       meta   = parseQAIRTMetadata(bundle);

    LLMModel m(buildSpecFromConfig(hf, meta));
    m.addInputProvider(makeEmbeddingProvider(hf, meta));
    m.addInputProvider(makeRoPEProvider(hf));
    return m;
}

inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) {
    try {
        LLMPipeline pipe;
        if (!pipe.create(chatMLTemplate, makeModel(model_cfg), runtime_cfg, model_cfg)) return std::nullopt;
        return pipe;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("qwen3::makePipeline failed: {}", e.what());
        return std::nullopt;
    }
}

}  // namespace qwen3
}  // namespace geniex
