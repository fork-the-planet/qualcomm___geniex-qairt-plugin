#pragma once

#include "llm/llm_types.h"
#include "llm/llm_model.h"
#include "llm/input_provider.h"
#include "pipeline/chat_template.h"
#include "pipeline/llm_pipeline.h"

namespace geniex {

// ── Granite4 chat template ──────────────────────────────────────────────────────────

// Granite4 role-tag format: <|start_of_role|>role<|end_of_role|>...<|end_of_text|>
inline std::string granite4ChatTemplate(const std::string& user_message,
                                        const std::string& system_prompt,
                                        bool first_turn, bool /*enable_thinking*/) {
    std::string result;
    if (first_turn && !system_prompt.empty())
        result += "<|start_of_role|>system<|end_of_role|>" + system_prompt +
                  "<|end_of_text|>\n";
    result += "<|start_of_role|>user<|end_of_role|>" + user_message +
              "<|end_of_text|>\n";
    result += "<|start_of_role|>assistant<|end_of_role|>";
    return result;
}

namespace granite4_micro {

static constexpr size_t  kHeadDim    = 64;
static constexpr float   kRopeTheta  = 10000000.0f;

// Returns the architecture spec for Granite4 Turbo (2 shards, multi-CL).
//
// Shard layout (same as granite4):
//   shard 0 : layers  0 – 19   – input_embeds → hidden
//   shard 1 : layers 20 – 39   – hidden → logits
inline LLMSpec makeSpec() {
    return LLMSpec{
        .shards = {
            {"input_embeds",         "_Add_99_Add_output_0"},
            {"_Add_99_Add_output_0", "logits"},
        },
        .state_blocks = {
            makeKVOnlyStateBlock({LayerRange{0, 19}, LayerRange{20, 39}}),
        },

        .seq_len_prefill = 128,
        .seq_len_decode  = 1,

        .hidden_size   = 2560,
        .num_heads     = 40,
        .num_kv_heads  = 8,
        .head_dim      = kHeadDim,
        .vocab_size    = 100352,

        .context_lengths = {1152, 4096},

        .eos_token_ids = {100257},
    };
}

// Returns a fully configured LLMModel with CPU-side embedding and RoPE providers.
inline LLMModel makeModel() {
    LLMModel m(makeSpec());
    m.addInputProvider(std::make_unique<EmbeddingInputProvider>());
    m.addInputProvider(std::make_unique<RoPEInputProvider>(kHeadDim, kRopeTheta));
    return m;
}

// Chat template used by this model family.
inline ChatTemplateFunc chatTemplate = granite4ChatTemplate;

// Creates a ready-to-use pipeline. Returns std::nullopt on failure.
inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg,
                                               const ModelConfig& model_cfg) {
    LLMPipeline pipe;
    if (!pipe.create(chatTemplate, makeModel(), runtime_cfg, model_cfg))
        return std::nullopt;
    return pipe;
}

} // namespace granite4_micro
} // namespace geniex
