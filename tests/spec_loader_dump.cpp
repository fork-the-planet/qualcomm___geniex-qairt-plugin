// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Verification harness: parses config.json + metadata.json from one or more
// bundle directories and prints the resulting LLMSpec / ParsedHFConfig /
// ParsedQAIRTMetadata. Read-only — does not load any QNN graphs.

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "llm/llm_spec_loader.h"

static const char* ropeTypeName(const geniex::RopeScaling& rs) {
    if (std::holds_alternative<geniex::Llama3RopeScaling>(rs)) return "llama3";
    if (std::holds_alternative<geniex::LongRopeScaling>(rs)) return "longrope";
    if (std::holds_alternative<geniex::PartialRopeScaling>(rs)) return "partial";
    return "standard";
}

static void dumpBundle(const std::filesystem::path& dir) {
    std::cout << "═══════════════════════════════════════════════════════════════\n"
              << "Bundle: " << dir.string() << "\n"
              << "═══════════════════════════════════════════════════════════════\n";
    try {
        auto hf   = geniex::parseHFConfig(dir);
        auto meta = geniex::parseQAIRTMetadata(dir);
        auto spec = geniex::buildSpecFromConfig(hf, meta);

        std::cout << "\n--- ParsedHFConfig (from config.json) ---\n"
                  << "  architecture            : " << hf.architecture << "\n"
                  << "  name_or_path            : " << hf.name_or_path << "\n"
                  << "  dialog_type             : " << geniex::parseGenieConfig(dir).dialog_type << "\n"
                  << "  model_type              : " << hf.model_type << "\n"
                  << "  hidden_size             : " << hf.hidden_size << "\n"
                  << "  num_attention_heads     : " << hf.num_attention_heads << "\n"
                  << "  num_key_value_heads     : " << hf.num_key_value_heads << "\n"
                  << "  head_dim                : " << hf.head_dim << "\n"
                  << "  vocab_size              : " << hf.vocab_size << "\n"
                  << "  num_hidden_layers       : " << hf.num_hidden_layers << "\n"
                  << "  max_position_embeddings : " << hf.max_position_embeddings << "\n"
                  << "  rope_theta              : " << hf.rope_theta << "\n"
                  << "  rope_scaling.type       : " << ropeTypeName(hf.rope_scaling) << "\n"
                  << "  bos_token_id            : " << hf.bos_token_id << "\n"
                  << "  pad_token_id            : " << hf.pad_token_id << "\n"
                  << "  eos_token_ids           : [";
        for (size_t i = 0; i < hf.eos_token_ids.size(); ++i) std::cout << (i ? ", " : "") << hf.eos_token_ids[i];
        std::cout << "]\n";
        if (!hf.mrope_section.empty()) {
            std::cout << "  mrope_section           : [";
            for (size_t i = 0; i < hf.mrope_section.size(); ++i) std::cout << (i ? ", " : "") << hf.mrope_section[i];
            std::cout << "]\n";
        }
        if (hf.image_token_id >= 0 || hf.vision_start_token_id >= 0 || hf.video_token_id >= 0) {
            std::cout << "  vision_start_token_id   : " << hf.vision_start_token_id << "\n"
                      << "  vision_end_token_id     : " << hf.vision_end_token_id << "\n"
                      << "  image_token_id          : " << hf.image_token_id << "\n"
                      << "  video_token_id          : " << hf.video_token_id << "\n";
        }

        std::cout << "\n--- ParsedQAIRTMetadata (from metadata.json) ---\n"
                  << "  shards (" << meta.shards.size() << "):\n";
        for (size_t s = 0; s < meta.shards.size(); ++s) {
            std::cout << "    [" << s + 1 << "] in='" << meta.shards[s].in_state_name << "'  out='"
                      << meta.shards[s].out_state_name << "'  lm_head_only=" << meta.shards[s].lm_head_only << "\n";
        }
        std::cout << "  shard_layer_ranges      : [";
        for (size_t s = 0; s < meta.shard_layer_ranges.size(); ++s) {
            if (s) std::cout << ", ";
            if (meta.shard_layer_ranges[s])
                std::cout << "[" << meta.shard_layer_ranges[s]->begin << ".." << meta.shard_layer_ranges[s]->end << "]";
            else
                std::cout << "<none>";
        }
        std::cout << "]\n";
        std::cout << "  context_lengths         : [";
        for (size_t i = 0; i < meta.context_lengths.size(); ++i)
            std::cout << (i ? ", " : "") << meta.context_lengths[i];
        std::cout << "]\n";
        std::cout << "  seq_len_prefill         : " << meta.seq_len_prefill << "\n"
                  << "  seq_len_decode          : " << meta.seq_len_decode << "\n"
                  << "  graph_name_pattern      : '" << meta.graph_name_pattern << "'\n"
                  << "  vision_encoder_graph    : '" << meta.vision_encoder_graph << "'\n";
        if (meta.vision_preprocessing) {
            const auto& vp = *meta.vision_preprocessing;
            std::cout << "  vision_preprocessing    :\n"
                      << "    image_width           : " << vp.image_width << "\n"
                      << "    image_height          : " << vp.image_height << "\n"
                      << "    patch_size            : " << vp.patch_size << "\n"
                      << "    temporal_patch_size   : " << vp.temporal_patch_size << "\n"
                      << "    spatial_merge_size    : " << vp.spatial_merge_size << "\n";
        }

        std::cout << "\n--- LLMSpec (composed) ---\n"
                  << "  shards.size             : " << spec.shards.size() << "\n"
                  << "  state_blocks.size       : " << spec.state_blocks.size() << "\n"
                  << "  hidden_size             : " << spec.hidden_size << "\n"
                  << "  num_heads               : " << spec.num_heads << "\n"
                  << "  num_kv_heads            : " << spec.num_kv_heads << "\n"
                  << "  head_dim                : " << spec.head_dim << "\n"
                  << "  vocab_size              : " << spec.vocab_size << "\n"
                  << "  seq_len_prefill         : " << spec.seq_len_prefill << "\n"
                  << "  seq_len_decode          : " << spec.seq_len_decode << "\n"
                  << "  context_lengths         : [";
        for (size_t i = 0; i < spec.context_lengths.size(); ++i)
            std::cout << (i ? ", " : "") << spec.context_lengths[i];
        std::cout << "]\n"
                  << "  attention_mask_name     : '" << spec.attention_mask_name << "'\n";

        std::cout << "\n[OK] " << dir.filename().string() << " parsed cleanly.\n";
    } catch (const std::exception& e) {
        std::cout << "\n[FAIL] " << e.what() << "\n";
    }
    std::cout << "\n";
}

int main(int argc, char** argv) {
    std::vector<std::filesystem::path> bundles;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) bundles.emplace_back(argv[i]);
    } else {
        const auto root = std::filesystem::current_path() / "modelfiles";
        bundles         = {
            root / "qwen3_4b",
            root / "qwen3_4b_instruct_2507",
            root / "qwen2_5_vl_7b",
        };
    }
    for (const auto& dir : bundles) dumpBundle(dir);
    return 0;
}
