#pragma once

#include "llm/llm_types.h"
#include "llm/llm_model.h"
#include "llm/input_provider.h"
#include "pipeline/chat_template.h"
#include "pipeline/llm_pipeline.h"

namespace geniex {
namespace qwen2_5_7b_instruct {

static constexpr size_t kHeadDim   = 128;
static constexpr float  kRopeTheta = 1000000.0f;

// Returns the architecture spec for Qwen2.5-7B-Instruct (6 shards).
//
// Shard layout:
//   shard 0 : embedding only   – input_ids → embeddings  (no KV cache)
//   shard 1 : layers  0 –  5   – embeddings → hidden
//   shard 2 : layers  6 – 11   – hidden → hidden
//   shard 3 : layers 12 – 17   – hidden → hidden
//   shard 4 : layers 18 – 23   – hidden → hidden
//   shard 5 : layers 24 – 27   – hidden → logits
inline LLMSpec makeSpec() {
    return LLMSpec{
        .shards = {
            {"input_ids",
             "_model_embed_tokens_Gather_Gather_output_0"},
            {"_model_embed_tokens_Gather_Gather_output_0",
             "_model_layers_5_Add_1_Add_output_0"},
            {"_model_layers_5_Add_1_Add_output_0",
             "_model_layers_11_Add_1_Add_output_0"},
            {"_model_layers_11_Add_1_Add_output_0",
             "_model_layers_17_Add_1_Add_output_0"},
            {"_model_layers_17_Add_1_Add_output_0",
             "_model_layers_23_Add_1_Add_output_0"},
            {"_model_layers_23_Add_1_Add_output_0",
             "logits"},
        },
        .state_blocks = {
            makeKVOnlyStateBlock({std::nullopt, LayerRange{0, 5}, LayerRange{6, 11}, LayerRange{12, 17}, LayerRange{18, 23}, LayerRange{24, 27}}),
        },

        .seq_len_prefill = 128,
        .seq_len_decode  = 128,

        .hidden_size  = 3584,
        .num_heads    = 28,
        .num_kv_heads = 4,
        .head_dim     = kHeadDim,
        .vocab_size   = 152064,

        .context_lengths = {4096},

        .graph_name_pattern = "{phase}_ar{ar}_cl{cl}_{shard}_of_{total}",

        .eos_token_ids = {151643, 151645},
    };
}

inline LLMModel makeModel() {
    LLMModel m(makeSpec());
    m.addInputProvider(std::make_unique<TokenIdInputProvider>("input_ids", 151643));
    m.addInputProvider(std::make_unique<RoPEInputProvider>(kHeadDim, kRopeTheta));
    return m;
}

inline ChatTemplateFunc chatTemplate = chatMLTemplate;

inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg,
                                               const ModelConfig& model_cfg) {
    LLMPipeline pipe;
    if (!pipe.create(chatTemplate, makeModel(), runtime_cfg, model_cfg))
        return std::nullopt;
    return pipe;
}

} // namespace qwen2_5_7b_instruct
} // namespace geniex
