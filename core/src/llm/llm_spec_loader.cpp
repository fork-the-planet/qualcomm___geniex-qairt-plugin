// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "llm/llm_spec_loader.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#include "logging.h"
#include "utils/detail/json.hpp"

namespace geniex {
namespace {

using json = qualla::json;

// Reads file into a json document; throws on missing file or parse error.
json loadJson(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("llm_spec_loader: cannot open " + path.string());
    }
    try {
        return json::parse(f);
    } catch (const std::exception& e) {
        throw std::runtime_error("llm_spec_loader: failed to parse " + path.string() + ": " + e.what());
    }
}

// Convert ONNX-style tensor name (slashes and dots) to the QNN runtime form
// used inside the compiled context binary (slashes → underscores, dots →
// underscores, leading slash kept as a leading underscore).
std::string canonicalTensorName(std::string name) {
    for (auto& c : name) {
        if (c == '/' || c == '.') c = '_';
    }
    return name;
}

bool isSpecialInput(const std::string& name) {
    if (name == "attention_mask") return true;
    if (name.rfind("position_ids", 0) == 0) return true;
    if (name.rfind("past_key_", 0) == 0) return true;
    if (name.rfind("past_value_", 0) == 0) return true;
    return false;
}

bool isSpecialOutput(const std::string& name) {
    if (name.rfind("past_key_", 0) == 0) return true;
    if (name.rfind("past_value_", 0) == 0) return true;
    return false;
}

// Captures (ar, cl, shard, total) and an optional phase prefix.
// Accepts forms with or without a leading phase prefix (prompt_/token_/prefill_/decode_).
struct GraphNameParts {
    std::string phase_prefix;  // empty, "prompt", "token", "prefill", "decode"
    size_t      ar    = 0;
    size_t      cl    = 0;
    size_t      shard = 0;  // 1-based
    size_t      total = 0;
};

bool parseGraphName(const std::string& name, GraphNameParts& out) {
    static const std::regex re(R"((?:([a-zA-Z]+)_)?ar(\d+)_cl(\d+)_(\d+)_of_(\d+))");
    std::smatch             m;
    if (!std::regex_match(name, m, re)) return false;
    out.phase_prefix = m[1].matched ? m[1].str() : "";
    out.ar           = std::stoul(m[2].str());
    out.cl           = std::stoul(m[3].str());
    out.shard        = std::stoul(m[4].str());
    out.total        = std::stoul(m[5].str());
    return true;
}

// Captures (shard, total) from `partN_of_M.bin`-style VLM keys.
// Accepts an optional leading prefix like `qwen2_5_vl_part1_of_5.bin`, which
// some bundles use, but the canonical schema is the bare `partN_of_M.bin`.
bool parsePartShardName(const std::string& name, size_t& shard, size_t& total) {
    static const std::regex re(R"((?:.*_)?part(\d+)_of_(\d+)\.bin)");
    std::smatch             m;
    if (!std::regex_match(name, m, re)) return false;
    shard = std::stoul(m[1].str());
    total = std::stoul(m[2].str());
    return true;
}

// Parses the integer N from "past_key_<N>_in" or "past_value_<N>_in".
// Returns std::nullopt on mismatch.
std::optional<size_t> parsePastIndex(const std::string& name) {
    static const std::regex re(R"(past_(?:key|value)_(\d+)_(?:in|out))");
    std::smatch             m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    return std::stoul(m[1].str());
}

template <typename T>
std::optional<T> getOpt(const json& j, const std::string& key) {
    if (!j.contains(key) || j.at(key).is_null()) return std::nullopt;
    return j.at(key).get<T>();
}

}  // namespace

ParsedHFConfig parseHFConfig(const std::filesystem::path& bundle_dir) {
    auto path = bundle_dir / "config.json";
    auto j    = loadJson(path);

    ParsedHFConfig cfg;
    if (j.contains("architectures") && j.at("architectures").is_array() && !j.at("architectures").empty()) {
        cfg.architecture = j.at("architectures").front().get<std::string>();
    }
    cfg.name_or_path        = j.value("_name_or_path", std::string{});
    cfg.model_type          = j.value("model_type", std::string{});
    cfg.hidden_size         = j.value("hidden_size", size_t{0});
    cfg.num_attention_heads = j.value("num_attention_heads", size_t{0});
    cfg.num_key_value_heads = j.value("num_key_value_heads", cfg.num_attention_heads);
    cfg.head_dim = j.value("head_dim", cfg.num_attention_heads ? cfg.hidden_size / cfg.num_attention_heads : size_t{0});
    cfg.vocab_size              = j.value("vocab_size", size_t{0});
    cfg.num_hidden_layers       = j.value("num_hidden_layers", size_t{0});
    cfg.max_position_embeddings = j.value("max_position_embeddings", size_t{0});
    cfg.rope_theta              = j.value("rope_theta", 10000.0f);

    // bos / pad token IDs
    if (auto v = getOpt<int32_t>(j, "bos_token_id")) cfg.bos_token_id = *v;
    if (auto v = getOpt<int32_t>(j, "pad_token_id")) cfg.pad_token_id = *v;

    // VLM-only token IDs (qwen2-vl-style configs). Absent or null on text-only models.
    if (auto v = getOpt<int32_t>(j, "vision_start_token_id")) cfg.vision_start_token_id = *v;
    if (auto v = getOpt<int32_t>(j, "vision_end_token_id")) cfg.vision_end_token_id = *v;
    if (auto v = getOpt<int32_t>(j, "image_token_id")) cfg.image_token_id = *v;
    if (auto v = getOpt<int32_t>(j, "video_token_id")) cfg.video_token_id = *v;

    // eos_token_id may be a single int or a list of ints.
    if (j.contains("eos_token_id") && !j.at("eos_token_id").is_null()) {
        const auto& eos = j.at("eos_token_id");
        if (eos.is_number_integer()) {
            cfg.eos_token_ids.push_back(eos.get<int32_t>());
        } else if (eos.is_array()) {
            for (const auto& e : eos) cfg.eos_token_ids.push_back(e.get<int32_t>());
        }
    }

    // RoPE scaling: a tagged HF object with "rope_type" or "type".
    if (j.contains("rope_scaling") && !j.at("rope_scaling").is_null() && j.at("rope_scaling").is_object()) {
        const auto& rs   = j.at("rope_scaling");
        std::string type = rs.value("rope_type", rs.value("type", std::string{}));
        if (type == "llama3") {
            Llama3RopeScaling s;
            s.factor                           = rs.value("factor", 1.0f);
            s.low_freq_factor                  = rs.value("low_freq_factor", 1.0f);
            s.high_freq_factor                 = rs.value("high_freq_factor", 4.0f);
            s.original_max_position_embeddings = rs.value("original_max_position_embeddings", size_t{8192});
            cfg.rope_scaling                   = s;
        } else if (type == "longrope" || type == "su") {
            LongRopeScaling s;
            if (rs.contains("long_factor"))
                for (const auto& v : rs.at("long_factor")) s.long_factor.push_back(v.get<float>());
            if (rs.contains("short_factor"))
                for (const auto& v : rs.at("short_factor")) s.short_factor.push_back(v.get<float>());
            s.original_max_position_embeddings = rs.value("original_max_position_embeddings", size_t{4096});
            cfg.rope_scaling                   = s;
        } else if (type == "partial") {
            PartialRopeScaling s;
            s.rope_fraction  = rs.value("rope_fraction", 1.0f);
            s.scale          = rs.value("scale", 1.0f);
            cfg.rope_scaling = s;
        } else {
            cfg.rope_scaling = StandardRope{};
        }
        // Qwen2-VL-style multi-dimensional RoPE: section sizes live alongside
        // the (default) rope_type. Captured separately so MRoPEInputProvider
        // can pick them up without changing the rope_scaling variant.
        if (rs.contains("mrope_section") && rs.at("mrope_section").is_array()) {
            for (const auto& v : rs.at("mrope_section")) cfg.mrope_section.push_back(v.get<int>());
        }
    } else {
        cfg.rope_scaling = StandardRope{};
    }

    return cfg;
}

// Inspects one graph entry and pulls out (in_state_name, out_state_name,
// kv_layer_indices). Throws if the entry is missing required tensors.
namespace {
struct ShardWiring {
    std::string      in_state;
    std::string      out_state;
    std::set<size_t> kv_layer_indices;
};

ShardWiring readShardWiring(const json& graph_entry, const std::string& diag_label) {
    ShardWiring w;
    if (graph_entry.contains("inputs") && graph_entry.at("inputs").is_object()) {
        for (auto it = graph_entry.at("inputs").begin(); it != graph_entry.at("inputs").end(); ++it) {
            const std::string& key = it.key();
            if (isSpecialInput(key)) {
                if (auto idx = parsePastIndex(key)) w.kv_layer_indices.insert(*idx);
                continue;
            }
            if (w.in_state.empty()) w.in_state = canonicalTensorName(key);
        }
    }
    if (graph_entry.contains("outputs") && graph_entry.at("outputs").is_object()) {
        for (auto it = graph_entry.at("outputs").begin(); it != graph_entry.at("outputs").end(); ++it) {
            const std::string& key = it.key();
            if (isSpecialOutput(key)) continue;
            if (w.out_state.empty()) w.out_state = canonicalTensorName(key);
        }
    }
    if (w.in_state.empty() || w.out_state.empty()) {
        throw std::runtime_error("llm_spec_loader: " + diag_label + " is missing a hidden-state input or output");
    }
    return w;
}

void parseVisionPreprocessing(const json& j, ParsedVisionPreprocessing& out) {
    out.image_width         = j.value("image_width", 0);
    out.image_height        = j.value("image_height", 0);
    out.patch_size          = j.value("patch_size", 0);
    out.temporal_patch_size = j.value("temporal_patch_size", 0);
    out.spatial_merge_size  = j.value("spatial_merge_size", 0);
    if (j.contains("normalize_mean") && j.at("normalize_mean").is_array()) {
        for (const auto& v : j.at("normalize_mean")) out.normalize_mean.push_back(v.get<float>());
    }
    if (j.contains("normalize_std") && j.at("normalize_std").is_array()) {
        for (const auto& v : j.at("normalize_std")) out.normalize_std.push_back(v.get<float>());
    }
}
}  // namespace

ParsedGenieConfig parseGenieConfig(const std::filesystem::path& bundle_dir) {
    ParsedGenieConfig out;
    auto              path = bundle_dir / "genie_config.json";
    if (!std::filesystem::exists(path)) return out;
    try {
        auto j = loadJson(path);
        if (j.contains("dialog") && j.at("dialog").is_object()) {
            const auto& dialog = j.at("dialog");
            if (dialog.contains("type") && dialog.at("type").is_string()) {
                out.dialog_type = dialog.at("type").get<std::string>();
            }
        }
    } catch (const std::exception& e) {
        GENIEX_LOG_WARN("llm_spec_loader: failed to parse genie_config.json: {}", e.what());
    }
    return out;
}

ParsedQAIRTMetadata parseQAIRTMetadata(const std::filesystem::path& bundle_dir) {
    auto path = bundle_dir / "metadata.json";
    auto j    = loadJson(path);

    if (!j.contains("model_files") || !j.at("model_files").is_object()) {
        throw std::runtime_error("llm_spec_loader: metadata.json missing 'model_files' object");
    }
    const auto& model_files = j.at("model_files");

    // First pass: classify each entry as either an LLM-style graph
    // (`ar<N>_cl<M>_<i>_of_<T>`) or a VLM-style shard (`partN_of_M.bin`).
    std::set<size_t>           ar_set;
    std::set<size_t>           cl_set;
    size_t                     total_shards_llm = 0;
    size_t                     total_shards_vlm = 0;
    std::optional<std::string> phase_prefix_seen;
    bool                       phase_prefix_consistent = true;
    std::string                vision_encoder_key;

    for (auto it = model_files.begin(); it != model_files.end(); ++it) {
        const std::string& key = it.key();
        GraphNameParts     parts;
        if (parseGraphName(key, parts)) {
            ar_set.insert(parts.ar);
            cl_set.insert(parts.cl);
            total_shards_llm = std::max(total_shards_llm, parts.total);
            if (!phase_prefix_seen) {
                phase_prefix_seen = parts.phase_prefix;
            } else if (*phase_prefix_seen != parts.phase_prefix) {
                phase_prefix_consistent = false;
            }
            continue;
        }
        size_t shard = 0, total = 0;
        if (parsePartShardName(key, shard, total)) {
            total_shards_vlm = std::max(total_shards_vlm, total);
            continue;
        }
        if (key == "vision_encoder.bin") {
            vision_encoder_key = key;
            continue;
        }
        GENIEX_LOG_WARN("llm_spec_loader: ignoring unrecognised graph entry '{}'", key);
    }

    const bool is_llm_style = total_shards_llm > 0;
    const bool is_vlm_style = total_shards_vlm > 0;
    if (!is_llm_style && !is_vlm_style) {
        throw std::runtime_error("llm_spec_loader: metadata.json contains no recognisable graph entries");
    }

    ParsedQAIRTMetadata out;
    out.vision_encoder_graph = vision_encoder_key;

    // vision_preprocessing block (VLM-only). Optional in any bundle, and may
    // sit at the top level (proposed schema) or under `genie` (Qwen2.5-VL).
    const json* vp_obj = nullptr;
    if (j.contains("vision_preprocessing") && j.at("vision_preprocessing").is_object()) {
        vp_obj = &j.at("vision_preprocessing");
    } else if (j.contains("genie") && j.at("genie").is_object() && j.at("genie").contains("vision_preprocessing") &&
               j.at("genie").at("vision_preprocessing").is_object()) {
        vp_obj = &j.at("genie").at("vision_preprocessing");
    }
    if (vp_obj) {
        ParsedVisionPreprocessing vp;
        parseVisionPreprocessing(*vp_obj, vp);
        out.vision_preprocessing = vp;
    }

    if (is_llm_style) {
        // ── LLM bundle: graphs keyed by ar<N>_cl<M>_<i>_of_<T> ────────────────
        out.context_lengths.assign(cl_set.begin(), cl_set.end());
        out.seq_len_prefill = *std::max_element(ar_set.begin(), ar_set.end());
        out.seq_len_decode  = *std::min_element(ar_set.begin(), ar_set.end());

        if (phase_prefix_consistent && phase_prefix_seen && !phase_prefix_seen->empty()) {
            out.graph_name_pattern = "{phase}_ar{ar}_cl{cl}_{shard}_of_{total}";
        } else {
            out.graph_name_pattern = "ar{ar}_cl{cl}_{shard}_of_{total}";
        }

        const size_t pref_ar = out.seq_len_prefill;
        const size_t pref_cl = out.context_lengths.front();

        out.shards.resize(total_shards_llm);
        out.shard_layer_ranges.assign(total_shards_llm, std::nullopt);

        for (size_t s = 1; s <= total_shards_llm; ++s) {
            std::vector<std::string> candidates;
            if (phase_prefix_seen && !phase_prefix_seen->empty()) {
                candidates.push_back(*phase_prefix_seen + "_ar" + std::to_string(pref_ar) + "_cl" +
                                     std::to_string(pref_cl) + "_" + std::to_string(s) + "_of_" +
                                     std::to_string(total_shards_llm));
            }
            candidates.push_back("ar" + std::to_string(pref_ar) + "_cl" + std::to_string(pref_cl) + "_" +
                                 std::to_string(s) + "_of_" + std::to_string(total_shards_llm));

            const json* graph_entry = nullptr;
            for (const auto& key : candidates) {
                if (model_files.contains(key)) {
                    graph_entry = &model_files.at(key);
                    break;
                }
            }
            if (!graph_entry) {
                for (auto it = model_files.begin(); it != model_files.end(); ++it) {
                    GraphNameParts parts;
                    if (parseGraphName(it.key(), parts) && parts.shard == s && parts.total == total_shards_llm) {
                        graph_entry = &it.value();
                        break;
                    }
                }
            }
            if (!graph_entry) {
                throw std::runtime_error(
                    "llm_spec_loader: could not locate graph entry for shard " + std::to_string(s));
            }

            auto      w = readShardWiring(*graph_entry, "shard " + std::to_string(s));
            ShardSpec sp;
            sp.in_state_name  = w.in_state;
            sp.out_state_name = w.out_state;
            sp.lm_head_only   = w.kv_layer_indices.empty() && w.out_state == "logits" && s > 1;
            out.shards[s - 1] = sp;

            if (!w.kv_layer_indices.empty()) {
                const size_t lo = *w.kv_layer_indices.begin();
                const size_t hi = *w.kv_layer_indices.rbegin();
                if (w.kv_layer_indices.size() != (hi - lo + 1)) {
                    throw std::runtime_error(
                        "llm_spec_loader: shard " + std::to_string(s) + " has non-contiguous KV layer indices");
                }
                out.shard_layer_ranges[s - 1] = LayerRange{lo, hi};
            }
        }
    } else {
        // ── VLM bundle: shards keyed by partN_of_M.bin, no AR/CL suffix ──────
        // Context lengths come from the top-level `context_lengths` array, or
        // from `genie.context_lengths` (Qwen2.5-VL convention).
        // AR is fixed by convention: prefill=128, decode=1.
        const json* cl_array = nullptr;
        if (j.contains("context_lengths") && j.at("context_lengths").is_array()) {
            cl_array = &j.at("context_lengths");
        } else if (j.contains("genie") && j.at("genie").is_object() && j.at("genie").contains("context_lengths") &&
                   j.at("genie").at("context_lengths").is_array()) {
            cl_array = &j.at("genie").at("context_lengths");
        }
        if (cl_array) {
            for (const auto& v : *cl_array) out.context_lengths.push_back(v.get<size_t>());
            std::sort(out.context_lengths.begin(), out.context_lengths.end());
        }
        if (out.context_lengths.empty()) {
            throw std::runtime_error(
                "llm_spec_loader: VLM-style metadata.json must declare top-level 'context_lengths'");
        }
        out.seq_len_prefill    = 128;
        out.seq_len_decode     = 1;
        out.graph_name_pattern = "";  // Bundles with bare partN_of_M keys: no name pattern.

        out.shards.resize(total_shards_vlm);
        out.shard_layer_ranges.assign(total_shards_vlm, std::nullopt);

        for (size_t s = 1; s <= total_shards_vlm; ++s) {
            const json* graph_entry = nullptr;
            for (auto it = model_files.begin(); it != model_files.end(); ++it) {
                size_t sh = 0, tot = 0;
                if (parsePartShardName(it.key(), sh, tot) && sh == s && tot == total_shards_vlm) {
                    graph_entry = &it.value();
                    break;
                }
            }
            if (!graph_entry) {
                throw std::runtime_error(
                    "llm_spec_loader: could not locate partN_of_M entry for shard " + std::to_string(s));
            }
            auto      w = readShardWiring(*graph_entry, "part" + std::to_string(s));
            ShardSpec sp;
            sp.in_state_name  = w.in_state;
            sp.out_state_name = w.out_state;
            sp.lm_head_only   = w.kv_layer_indices.empty() && w.out_state == "logits" && s > 1;
            out.shards[s - 1] = sp;

            if (!w.kv_layer_indices.empty()) {
                const size_t lo = *w.kv_layer_indices.begin();
                const size_t hi = *w.kv_layer_indices.rbegin();
                if (w.kv_layer_indices.size() != (hi - lo + 1)) {
                    throw std::runtime_error(
                        "llm_spec_loader: part" + std::to_string(s) + " has non-contiguous KV layer indices");
                }
                out.shard_layer_ranges[s - 1] = LayerRange{lo, hi};
            }
        }
    }

    return out;
}

LLMSpec buildSpecFromConfig(const ParsedHFConfig& hf, const ParsedQAIRTMetadata& meta) {
    LLMSpec spec;
    spec.shards = meta.shards;

    spec.state_blocks = {makeKVOnlyStateBlock(meta.shard_layer_ranges)};

    spec.seq_len_prefill    = meta.seq_len_prefill;
    spec.seq_len_decode     = meta.seq_len_decode;
    spec.hidden_size        = hf.hidden_size;
    spec.num_heads          = hf.num_attention_heads;
    spec.num_kv_heads       = hf.num_key_value_heads;
    spec.head_dim           = hf.head_dim;
    spec.vocab_size         = hf.vocab_size;
    spec.context_lengths    = meta.context_lengths;
    spec.graph_name_pattern = meta.graph_name_pattern;
    spec.eos_token_ids      = hf.eos_token_ids;
    return spec;
}

std::unique_ptr<InputProvider> makeRoPEProvider(const ParsedHFConfig& hf) {
    return std::visit(
        [&](const auto& s) -> std::unique_ptr<InputProvider> {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, Llama3RopeScaling>) {
                // No dedicated Llama3RoPEInputProvider in core today; the
                // standard provider matches all currently shipped llama3 bundles
                // (which export pre-baked RoPE tables anyway). Log so it's
                // visible if a future bundle starts depending on the scaling.
                GENIEX_LOG_INFO(
                    "llm_spec_loader: rope_scaling=llama3 (factor={}); using standard RoPE provider", s.factor);
                return std::make_unique<RoPEInputProvider>(hf.head_dim, hf.rope_theta);
            } else if constexpr (std::is_same_v<T, LongRopeScaling>) {
                const size_t orig = s.original_max_position_embeddings ? s.original_max_position_embeddings : 4096;
                const size_t maxp = hf.max_position_embeddings ? hf.max_position_embeddings : 131072;
                return std::make_unique<LongRoPEInputProvider>(
                    hf.head_dim, hf.rope_theta, s.long_factor, static_cast<int>(maxp), static_cast<int>(orig));
            } else if constexpr (std::is_same_v<T, PartialRopeScaling>) {
                return std::make_unique<PartialRoPEInputProvider>(hf.head_dim, hf.rope_theta, s.rope_fraction, s.scale);
            } else {
                return std::make_unique<RoPEInputProvider>(hf.head_dim, hf.rope_theta);
            }
        },
        hf.rope_scaling);
}

std::unique_ptr<InputProvider> makeEmbeddingProvider(const ParsedHFConfig& hf, const ParsedQAIRTMetadata& meta) {
    if (meta.shards.empty()) {
        throw std::runtime_error("llm_spec_loader: cannot pick embedding provider — no shards parsed");
    }
    const std::string& first = meta.shards.front().in_state_name;
    if (first == "input_ids") {
        // Pad token: prefer hf.pad_token_id, then first eos, then 0.
        int32_t pad = hf.pad_token_id;
        if (pad < 0) pad = hf.eos_token_ids.empty() ? 0 : hf.eos_token_ids.front();
        return std::make_unique<TokenIdInputProvider>("input_ids", pad);
    }
    if (first == "input_embeds" || first == "inputs_embeds") {
        return std::make_unique<EmbeddingInputProvider>(first);
    }
    throw std::runtime_error("llm_spec_loader: unrecognised first-shard input '" + first +
                             "' — expected 'input_ids', 'input_embeds', or 'inputs_embeds'");
}

std::filesystem::path bundleDirOf(const ModelConfig& model_cfg) {
    if (model_cfg.model_paths.empty()) {
        throw std::runtime_error("llm_spec_loader: model_cfg.model_paths is empty");
    }
    return std::filesystem::path(model_cfg.model_paths.front()).parent_path();
}

ModelConfig modelConfigFromDirectory(const std::filesystem::path& bundle_dir) {
    ModelConfig cfg;

    // Tokenizer.
    auto tok = bundle_dir / "tokenizer.json";
    if (!std::filesystem::exists(tok)) {
        throw std::runtime_error("llm_spec_loader: tokenizer.json not found in " + bundle_dir.string());
    }
    cfg.tokenizer_path = tok.string();

    // HTP backend extension config (optional).
    auto htp = bundle_dir / "htp_backend_ext_config.json";
    if (std::filesystem::exists(htp)) cfg.htp_config_path = htp.string();

    // Context binaries: prefer genie_config.json's ctx-bins ordering.
    auto genie_path = bundle_dir / "genie_config.json";
    if (std::filesystem::exists(genie_path)) {
        try {
            auto gj = loadJson(genie_path);
            if (gj.contains("dialog") && gj.at("dialog").contains("engine") &&
                gj.at("dialog").at("engine").contains("model") &&
                gj.at("dialog").at("engine").at("model").contains("binary") &&
                gj.at("dialog").at("engine").at("model").at("binary").contains("ctx-bins")) {
                for (const auto& b : gj.at("dialog").at("engine").at("model").at("binary").at("ctx-bins")) {
                    cfg.model_paths.push_back((bundle_dir / b.get<std::string>()).string());
                }
            }
        } catch (const std::exception& e) {
            GENIEX_LOG_WARN("llm_spec_loader: failed to read genie_config.json: {}", e.what());
        }
    }

    // Fallback: glob the directory for *.bin and sort lexicographically.
    if (cfg.model_paths.empty()) {
        std::vector<std::string> bins;
        for (const auto& entry : std::filesystem::directory_iterator(bundle_dir)) {
            if (entry.path().extension() == ".bin") bins.push_back(entry.path().string());
        }
        std::sort(bins.begin(), bins.end());
        cfg.model_paths = std::move(bins);
    }

    if (cfg.model_paths.empty()) {
        throw std::runtime_error("llm_spec_loader: no .bin shards found in " + bundle_dir.string());
    }
    return cfg;
}

}  // namespace geniex
