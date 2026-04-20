#pragma once

#include "llm/llm_types.h"
#include "llm/llm_model.h"
#include "llm/input_provider.h"
#include "pipeline/chat_template.h"
#include "pipeline/llm_pipeline.h"

namespace geniex {
namespace phi4 {

static constexpr size_t  kHeadDim    = 128;
static constexpr float   kRopeTheta  = 10000.0f;
static constexpr float   kRopeScale  = 1.19023807f;

// Returns the architecture spec for Phi4-mini Turbo (2 shards, multi-CL).
inline LLMSpec makeSpec() {
    return LLMSpec{
        .shards = {
            {"input_embeds",         "_Add_78_Add_output_0"},
            {"_Add_78_Add_output_0", "logits"},
        },
        .state_blocks = {
            makeKVOnlyStateBlock({LayerRange{0, 15}, LayerRange{16, 31}}),
        },

        .seq_len_prefill = 16,
        .seq_len_decode  = 1,

        .hidden_size   = 3072,
        .num_heads     = 24,
        .num_kv_heads  = 8,
        .head_dim      = kHeadDim,
        .vocab_size    = 200064,

        .context_lengths = {640, 1152, 2176, 4096},

        .eos_token_ids = {200020},
    };
}

inline LLMModel makeModel() {
    LLMModel m(makeSpec());
    m.addInputProvider(std::make_unique<EmbeddingInputProvider>());
    m.addInputProvider(std::make_unique<PartialRoPEInputProvider>(kHeadDim, kRopeTheta, 0.75f, kRopeScale));
    return m;
}

inline ChatTemplateFunc chatTemplate = phiChatTemplate;

inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg,
                                               const ModelConfig& model_cfg) {
    LLMPipeline pipe;
    if (!pipe.create(chatTemplate, makeModel(), runtime_cfg, model_cfg))
        return std::nullopt;
    return pipe;
}

} // namespace phi4_turbo
} // namespace geniex
