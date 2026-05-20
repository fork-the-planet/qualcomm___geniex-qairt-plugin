// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "geniex_export.h"
#include "llm/input_provider.h"
#include "llm/llm_types.h"
#include "types.h"

// Reads a QAIRT distributed model bundle (config.json + metadata.json) and
// produces an LLMSpec + matching CPU-side InputProviders, so that family
// runtimes do not need to hardcode shape/wiring constants.
//
// Bundle layout (matches modelfiles/qwen3_4b_instruct_2507/):
//   config.json              — HuggingFace-style transformer config
//   metadata.json            — QAIRT export metadata (per-graph I/O tensors)
//   genie_config.json        — optional; lists ctx-bins (used by directory helper)
//   tokenizer.json           — sentencepiece/BPE tokenizer
//   htp_backend_ext_config.json — HTP backend extension config (optional)
//   *.bin                    — compiled context binary shards

namespace geniex {

// RoPE scaling variants supported in HF config.json's "rope_scaling" field.
struct StandardRope {};

struct Llama3RopeScaling {
    float  factor;
    float  low_freq_factor;
    float  high_freq_factor;
    size_t original_max_position_embeddings;
};

struct LongRopeScaling {
    std::vector<float> long_factor;
    std::vector<float> short_factor;
    size_t             original_max_position_embeddings;
};

struct PartialRopeScaling {
    float rope_fraction;
    float scale;
};

using RopeScaling = std::variant<StandardRope, Llama3RopeScaling, LongRopeScaling, PartialRopeScaling>;

// Parsed HuggingFace config.json. Only the fields required by buildSpecFromConfig
// and the input-provider factories are kept.
struct ParsedHFConfig {
    // First entry of config.json's "architectures" array (e.g. "Qwen3ForCausalLM",
    // "LlamaForCausalLM", "Phi3ForCausalLM", "Qwen2_5_VLForConditionalGeneration").
    // Empty when absent. Note: Falcon3 also reports "LlamaForCausalLM" — use
    // `name_or_path` to disambiguate.
    std::string architecture;

    // config.json's "_name_or_path" — the original HuggingFace repo path
    // (e.g. "meta-llama/Llama-3.2-3B-Instruct", "tiiuae/Falcon3-7B-Instruct").
    // Used to disambiguate same-architecture families (Falcon3 vs Llama).
    std::string name_or_path;

    std::string model_type;          // e.g. "qwen3", "llama", "phi3", "qwen2_5_vl_text"
    size_t      hidden_size         = 0;
    size_t      num_attention_heads = 0;
    size_t      num_key_value_heads = 0;
    size_t      head_dim            = 0;
    size_t      vocab_size          = 0;
    size_t      num_hidden_layers   = 0;
    size_t      max_position_embeddings = 0;

    float       rope_theta = 10000.0f;
    RopeScaling rope_scaling = StandardRope{};

    int32_t              bos_token_id = -1;
    int32_t              pad_token_id = -1;
    std::vector<int32_t> eos_token_ids;

    // VLM-only fields. Populated when present in config.json (HF qwen2-vl style):
    //   rope_scaling.mrope_section
    //   vision_start_token_id / vision_end_token_id / image_token_id / video_token_id
    // Empty / -1 when not applicable.
    std::vector<int>  mrope_section;
    int32_t           vision_start_token_id = -1;
    int32_t           vision_end_token_id   = -1;
    int32_t           image_token_id        = -1;
    int32_t           video_token_id        = -1;
};

// Vision preprocessing knobs read from metadata.json's `vision_preprocessing`
// block. All fields are zero when the bundle is not a VLM.
struct ParsedVisionPreprocessing {
    int                image_width         = 0;
    int                image_height        = 0;
    int                patch_size          = 0;
    int                temporal_patch_size = 0;
    int                spatial_merge_size  = 0;
    std::vector<float> normalize_mean;
    std::vector<float> normalize_std;
};

// Parsed QAIRT metadata.json. Captures only the runtime-wiring fields.
struct ParsedQAIRTMetadata {
    // Per-shard hidden-state wiring (in_state_name / out_state_name).
    std::vector<ShardSpec> shards;

    // Per-shard inclusive layer ranges (one std::optional per shard; nullopt for
    // shards without KV state, e.g. an embedding-only or lm-head-only shard).
    std::vector<std::optional<LayerRange>> shard_layer_ranges;

    // Distinct context lengths the export supports (sorted ascending).
    // Sourced from `cl<N>` graph-name suffixes (LLM bundles) or from the
    // top-level `context_lengths` array (VLM bundles).
    std::vector<size_t> context_lengths;

    size_t seq_len_prefill = 0;
    size_t seq_len_decode  = 0;

    // Pattern recovered from the graph names. Empty for VLM bundles, where the
    // metadata uses bare `partN_of_M.bin` keys with no AR/CL suffix.
    std::string graph_name_pattern;

    // Optional vision-preprocessing block (VLM bundles only).
    std::optional<ParsedVisionPreprocessing> vision_preprocessing;

    // Optional vision-encoder graph entry (VLM bundles). Empty string means
    // the bundle has no separate vision graph.
    std::string vision_encoder_graph;
};

// Subset of `genie_config.json` we care about for runtime dispatch.
// Other fields (sampler defaults, HTP tuning, embedding LUT path) can be
// added here as they become useful.
struct ParsedGenieConfig {
    // `dialog.type` — selects the decoding strategy. Known values:
    //   "basic"        — standard LLM (default if file is absent)
    //   "ssd-q1"       — Self-Speculative Decoding
    //   "spd"          — Speculative Decoding
    //   "lade"         — Lookahead Decoding
    //   "kv-share"     — KV-share multi-engine
    //   "multistream"  — Multi-stream
    //   "eaglet"       — EAGLE-style speculation
    std::string dialog_type = "basic";
};

// Reads HuggingFace-style config.json from the bundle directory.
// Throws std::runtime_error on missing / malformed config.
GENIEX_API ParsedHFConfig parseHFConfig(const std::filesystem::path& bundle_dir);

// Reads genie_config.json. Returns an all-defaults struct if the file is
// absent (most modelfile bundles ship one, but it's not strictly required).
GENIEX_API ParsedGenieConfig parseGenieConfig(const std::filesystem::path& bundle_dir);

// Reads QAIRT metadata.json from the bundle directory.
// Throws std::runtime_error on missing / malformed metadata.
GENIEX_API ParsedQAIRTMetadata parseQAIRTMetadata(const std::filesystem::path& bundle_dir);

// Combines the two parsed structures into a fully-populated LLMSpec.
GENIEX_API LLMSpec buildSpecFromConfig(const ParsedHFConfig& hf, const ParsedQAIRTMetadata& meta);

// Picks the matching RoPE input provider implementation from `hf.rope_scaling`.
// Falls back to the standard provider if the scaling variant is unrecognised.
GENIEX_API std::unique_ptr<InputProvider> makeRoPEProvider(const ParsedHFConfig& hf);

// Picks the embedding-input provider for shard 0 based on its expected input
// tensor name. Returns a TokenIdInputProvider when the first shard takes
// "input_ids" (on-device embedding) or an EmbeddingInputProvider when it
// takes "input_embeds" / "inputs_embeds" (CPU-side embedding lookup).
GENIEX_API std::unique_ptr<InputProvider> makeEmbeddingProvider(
    const ParsedHFConfig& hf, const ParsedQAIRTMetadata& meta);

// Returns the directory that contains the modelfile bundle for `model_cfg`.
// Inferred as the parent directory of model_cfg.model_paths[0].
GENIEX_API std::filesystem::path bundleDirOf(const ModelConfig& model_cfg);

// Convenience: derive a ModelConfig from a bundle directory by reading
// genie_config.json (for ctx-bins ordering), tokenizer.json, and
// htp_backend_ext_config.json. Used by example executables so they only
// have to know the bundle directory.
GENIEX_API ModelConfig modelConfigFromDirectory(const std::filesystem::path& bundle_dir);

}  // namespace geniex
