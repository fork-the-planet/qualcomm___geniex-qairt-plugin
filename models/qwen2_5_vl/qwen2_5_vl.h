#pragma once

#include "llm/input_provider.h"
#include "llm/llm_model.h"
#include "llm/llm_types.h"
#include "pipeline/chat_template.h"
#include "pipeline/llm_pipeline.h"

namespace geniex {
namespace qwen2_5_vl_7b {

// ── Architecture constants ────────────────────────────────────────────────────
// Source of truth: modelfiles/qwen2_5_vl/config.json
static constexpr size_t kHiddenSize   = 3584;
static constexpr size_t kNumHeads     = 28;
static constexpr size_t kNumKVHeads   = 4;
static constexpr size_t kHeadDim      = 128;   // hidden_size / num_heads
static constexpr size_t kNumLayers    = 28;
static constexpr size_t kVocabSize    = 152064;
static constexpr float  kRopeTheta    = 1000000.0f;

// Returns the architecture spec for the Qwen2.5-VL-7B text backbone
// (5-shard Genie export, CL=2048, prefill AR=128, decode AR=1).
//
// Shard layout (28 transformer layers spread across 5 shards, no embed shard):
//   shard 1 : layers  0 –  5   – inputs_embeds → add_13335
//   shard 2 : layers  6 – 11   – add_13335     → add_25971
//   shard 3 : layers 12 – 17   – add_25971     → add_38607
//   shard 4 : layers 18 – 23   – add_38607     → add_51243
//   shard 5 : layers 24 – 27   – add_51243     → logits
//
// Notes:
//  • The embedding lookup is done CPU-side (no on-device embed shard), using
//    the raw fp32 table in modelfiles/qwen2_5_vl/embedding_weights.raw.
//  • The full Qwen2.5-VL model uses MRoPE with a 3-dim position axis. For
//    text-only prompts the three axes collapse to the usual 1-D position IDs,
//    so the standard RoPEInputProvider is sufficient — the exported graph
//    takes cos/sin tensors of shape [1, 1, AR, head_dim/2] which match.
//    Vision/video inputs will later need a dedicated MRoPE provider.
inline LLMSpec makeSpec() {
    return LLMSpec{
        .shards = {
            {"inputs_embeds", "add_13335"},
            {"add_13335",     "add_25971"},
            {"add_25971",     "add_38607"},
            {"add_38607",     "add_51243"},
            {"add_51243",     "logits"},
        },
        .state_blocks = {
            makeKVOnlyStateBlock({
                LayerRange{0,  5},
                LayerRange{6,  11},
                LayerRange{12, 17},
                LayerRange{18, 23},
                LayerRange{24, 27},
            }),
        },

        .seq_len_prefill = 128,
        .seq_len_decode  = 1,

        .hidden_size  = kHiddenSize,
        .num_heads    = kNumHeads,
        .num_kv_heads = kNumKVHeads,
        .head_dim     = kHeadDim,
        .vocab_size   = kVocabSize,

        .context_lengths = {2048},

        // Genie export graph names: prompt_arAR_clCL_N_of_TOTAL / token_...
        .graph_name_pattern = "{phase}_ar{ar}_cl{cl}_{shard}_of_{total}",

        // eos (<|im_end|> = 151645) + endoftext (151643) fallback.
        .eos_token_ids = {151643, 151645},
    };
}

// Builds an LLMModel configured for text-only inference.
// The caller must set `model_cfg.embedding_path` to the raw fp32 embedding
// table (vocab_size × hidden_size floats, row-major) before initialize().
inline LLMModel makeModel() {
    LLMModel m(makeSpec());

    // Shard-1 input tensor is named "inputs_embeds" (note the 's'). The
    // provider loads the embedding table CPU-side from model_cfg.embedding_path
    // and auto-detects the format:
    //   * .npy     — shape read from the numpy header
    //   * raw .raw — shape inferred from the shard-1 tensor spec + file size
    m.addInputProvider(std::make_unique<EmbeddingInputProvider>("inputs_embeds"));

    // Text-only path: standard RoPE. Writes "position_ids_cos" / "position_ids_sin".
    m.addInputProvider(std::make_unique<RoPEInputProvider>(kHeadDim, kRopeTheta));

    return m;
}

// Qwen2.5-VL uses the ChatML template (same as Qwen2.5 / Qwen3).
inline ChatTemplateFunc chatTemplate = chatMLTemplate;

// Factory: builds an LLMPipeline ready for generate().
inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg,
                                               const ModelConfig&      model_cfg) {
    LLMPipeline pipe;
    if (!pipe.create(chatTemplate, makeModel(), runtime_cfg, model_cfg))
        return std::nullopt;
    return pipe;
}

} // namespace qwen2_5_vl_7b
} // namespace geniex
