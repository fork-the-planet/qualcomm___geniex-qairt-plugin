#pragma once

#include "llm/llm_types.h"
#include "llm/llm_model.h"
#include "llm/input_provider.h"
#include "pipeline/chat_template.h"
#include "pipeline/llm_pipeline.h"

namespace geniex {
namespace phi3_5 {

static constexpr size_t  kHeadDim    = 96;
static constexpr float   kRopeTheta  = 10000.0f;

// Per-dimension LongRoPE extension factors (48 values for head_dim=96).
// Source: Phi3.5-mini model tensor.
static inline const std::vector<float> kExtFactors = {
    1.0000f, 1.0200f, 1.0300f, 1.0300f, 1.0500f, 1.0500f, 1.0500f, 1.0500f, 1.0500f,
    1.0700f, 1.1000f, 1.1100f, 1.1600f, 1.1600f, 1.1700f, 1.2900f, 1.3400f, 1.6800f,
    1.7900f, 1.8200f, 1.8500f, 1.8800f, 1.9100f, 1.9400f, 1.9900f, 2.0200f, 2.0200f,
    2.0200f, 2.0200f, 2.0200f, 2.0200f, 2.0300f, 2.0300f, 2.0300f, 2.0300f, 2.0300f,
    2.0300f, 2.0300f, 2.0300f, 2.0300f, 2.0800f, 2.0900f, 2.1900f, 2.2200f, 2.5900f,
    2.7300f, 2.7500f, 2.8400f
};

// Returns the architecture spec for Phi3.5-mini (2 shards, CPU embedding).
//
// Shard layout:
//   shard 0 : layers  0 – 15   – input_embeds → _Add_78_Add_output_0  (KV layers 0–15)
//   shard 1 : layers 16 – 31   – _Add_78_Add_output_0 → logits        (KV layers 16–31)
inline LLMSpec makeSpec() {
    return LLMSpec{
        .shards = {
            {"input_embeds",         "_Add_78_Add_output_0"},
            {"_Add_78_Add_output_0", "logits"},
        },
        .state_blocks = {
            makeKVOnlyStateBlock({LayerRange{0, 15}, LayerRange{16, 31}}),
        },

        .seq_len_prefill = 32,
        .seq_len_decode  = 1,

        .hidden_size   = 3072,
        .num_heads     = 32,
        .num_kv_heads  = 32,
        .head_dim      = kHeadDim,
        .vocab_size    = 32064,

        .context_lengths = {1536},

        .eos_token_ids = {32007, 32000},
    };
}

// Returns a fully configured LLMModel with CPU-side embedding and Phi3.5 RoPE providers.
inline LLMModel makeModel() {
    LLMModel m(makeSpec());
    m.addInputProvider(std::make_unique<EmbeddingInputProvider>());
    m.addInputProvider(std::make_unique<LongRoPEInputProvider>(kHeadDim, kRopeTheta, kExtFactors));
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

} // namespace phi3_5

namespace phi3_5_aihub {

static constexpr size_t  kHeadDim    = 96;
static constexpr float   kRopeTheta  = 10000.0f;

// Per-dimension LongRoPE extension factors (48 values for head_dim=96).
// Source: Phi3.5-mini model tensor.
static inline const std::vector<float> kExtFactors = {
    1.0000f, 1.0200f, 1.0300f, 1.0300f, 1.0500f, 1.0500f, 1.0500f, 1.0500f, 1.0500f,
    1.0700f, 1.1000f, 1.1100f, 1.1600f, 1.1600f, 1.1700f, 1.2900f, 1.3400f, 1.6800f,
    1.7900f, 1.8200f, 1.8500f, 1.8800f, 1.9100f, 1.9400f, 1.9900f, 2.0200f, 2.0200f,
    2.0200f, 2.0200f, 2.0200f, 2.0200f, 2.0300f, 2.0300f, 2.0300f, 2.0300f, 2.0300f,
    2.0300f, 2.0300f, 2.0300f, 2.0300f, 2.0800f, 2.0900f, 2.1900f, 2.2200f, 2.5900f,
    2.7300f, 2.7500f, 2.8400f
};

// Returns the architecture spec for Phi3.5-mini AI Hub export (4 shards, 5 CL variants).
//
// Shard layout (per the AI Hub export):
//   shard 0 : embedding only   – input_ids → embeddings           (no KV cache)
//   shard 1 : layers  0 – 15   – embeddings → hidden              (KV layers 0–15)
//   shard 2 : layers 16 – 31   – hidden → hidden                  (KV layers 16–31)
//   shard 3 : lm_head only     – hidden → logits                  (no KV cache)
inline LLMSpec makeSpec() {
    return LLMSpec{
        .shards = {
            {"input_ids",
             "_model_embed_tokens_Gather_Gather_output_0"},
            {"_model_embed_tokens_Gather_Gather_output_0",
             "_model_layers_15_Add_1_Add_output_0"},
            {"_model_layers_15_Add_1_Add_output_0",
             "_model_layers_31_Add_1_Add_output_0"},
            {"_model_layers_31_Add_1_Add_output_0",
             "logits"},
        },
        .state_blocks = {
            makeKVOnlyStateBlock({std::nullopt, LayerRange{0, 15}, LayerRange{16, 31}, std::nullopt}),
        },

        .seq_len_prefill = 128,
        .seq_len_decode  = 1,

        .hidden_size   = 3072,
        .num_heads     = 32,
        .num_kv_heads  = 32,
        .head_dim      = kHeadDim,
        .vocab_size    = 32064,

        .context_lengths = {512, 1024, 2048, 3072, 4096},

        // Default graph_name_pattern matches: ar128_cl4096_1_of_4, ar1_cl4096_1_of_4, etc.

        .eos_token_ids = {32007, 32000},
    };
}

// Returns a fully configured LLMModel with on-device embedding and Phi3.5 RoPE providers.
// No CPU-side embedding table needed – shard 0 does embedding on-device.
inline LLMModel makeModel() {
    LLMModel m(makeSpec());
    m.addInputProvider(std::make_unique<TokenIdInputProvider>("input_ids", 0));
    m.addInputProvider(std::make_unique<LongRoPEInputProvider>(kHeadDim, kRopeTheta, kExtFactors));
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

} // namespace phi3_5_aihub
} // namespace geniex
