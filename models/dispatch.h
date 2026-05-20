// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Architecture-driven pipeline dispatcher.
//
// Replaces the per-variant llm_model_registry / vlm_model_registry tables: the
// runtime reads the bundle's standard QAIRT distribution files and routes to
// the matching family factory. Adding a new variant of an existing
// architecture requires no source change — drop the bundle into
// modelfiles/<name>/ and call makeLLMPipeline / makeVLMPipeline.
//
// Dispatch rules (in priority order):
//   genie_config.json `dialog.type`              architecture                       → factory
//   ─────────────────────────────────────────────────────────────────────────────────────────────────────
//   "ssd-q1"                                     LlamaForCausalLM                   → llama3_2_3b_ssd::makePipeline
//   "basic" (or absent)                          LlamaForCausalLM, "falcon" in path → falcon3::makePipeline
//   "basic" (or absent)                          LlamaForCausalLM                   → llama3::makePipeline
//   "basic" (or absent)                          Qwen3ForCausalLM                   → qwen3::makePipeline
//   "basic" (or absent)                          Qwen2ForCausalLM                   → qwen2_5::makePipeline
//   "basic" (or absent)                          Phi3ForCausalLM / Phi3VForCausalLM → phi3_5::makePipeline
//   (n/a)                                        Qwen2_5_VLForConditionalGeneration → qwen2_5_vl::makePipeline (VLM)
//
// Why two signals:
//   - `architectures[0]` is set by HuggingFace `transformers` and identifies the
//     model class (Llama vs Qwen vs Phi vs VL). It alone can't distinguish:
//       * SSD vs plain Llama (same arch, same shapes)
//       * Falcon3 vs Llama-3 (Falcon3 reports architectures=["LlamaForCausalLM"])
//   - `genie_config.json`'s `dialog.type` (the same field Genie itself uses for
//     this decision in Genie/src/Dialog.cpp) declares the decoding strategy
//     authoritatively. We use it for SSD; Genie uses it for SPD/LADE/etc. too.
//   - `_name_or_path` ("tiiuae/Falcon3-…" vs "meta-llama/Llama-…") is the only
//     in-bundle signal that distinguishes Falcon3 from Llama-3.
//
// SSD runtime detail: when dispatching to SSD, we auto-discover the forecast
// prefix file under `<bundle>/forecast-prefix/` if the caller did not set
// model_cfg.forecast_prefix_path explicitly.

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <optional>
#include <string>

#include "llm/llm_spec_loader.h"
#include "logging.h"
#include "pipeline/llm_pipeline.h"
#include "pipeline/vlm_pipeline.h"
#include "types.h"

#include "falcon3/falcon3.h"
#include "llama3/llama3.h"
#include "llama3_2_ssd/llama3_2_ssd.h"
#include "phi3_5/phi3_5.h"
#include "qwen2_5/qwen2_5.h"
#include "qwen2_5_vl/qwen2_5_vl.h"
#include "qwen3/qwen3.h"

namespace geniex {

namespace dispatch_detail {

inline std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

inline bool containsCaseInsensitive(const std::string& haystack, const std::string& needle) {
    return toLower(haystack).find(toLower(needle)) != std::string::npos;
}

// Combined signature of the bundle: HF architecture + HF name_or_path + Genie
// dialog type. All read at dispatch time.
struct BundleSignature {
    std::string architecture;   // config.json `architectures[0]`
    std::string name_or_path;   // config.json `_name_or_path`
    std::string dialog_type;    // genie_config.json `dialog.type`, defaults to "basic"
};

inline BundleSignature signatureOf(const ModelConfig& model_cfg) {
    BundleSignature sig;
    try {
        const auto bundle = bundleDirOf(model_cfg);
        auto       hf     = parseHFConfig(bundle);
        auto       gc     = parseGenieConfig(bundle);
        sig.architecture  = hf.architecture;
        sig.name_or_path  = hf.name_or_path;
        sig.dialog_type   = gc.dialog_type;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("dispatch: cannot read config.json: {}", e.what());
    }
    return sig;
}

// If `forecast_prefix_path` is unset and the bundle contains a
// `forecast-prefix/kv-cache.primary.qnn-htp` file, populate it. Returns the
// (possibly mutated) ModelConfig.
inline ModelConfig autoDiscoverForecastPrefix(ModelConfig model_cfg) {
    if (model_cfg.forecast_prefix_path.has_value()) return model_cfg;
    try {
        const auto bundle = bundleDirOf(model_cfg);
        const auto candidate = bundle / "forecast-prefix" / "kv-cache.primary.qnn-htp";
        if (std::filesystem::exists(candidate)) {
            model_cfg.forecast_prefix_path = candidate.string();
        }
    } catch (...) {
        // bundleDirOf may throw if model_paths is empty; nothing to autodiscover.
    }
    return model_cfg;
}

}  // namespace dispatch_detail

// Single LLM entry point. Dispatches to the correct family factory based on
// the bundle's standard QAIRT distribution files (config.json + genie_config.json).
// Returns std::nullopt for unknown / VLM architectures.
inline std::optional<LLMPipeline> makeLLMPipeline(const QnnRuntimeConfig& runtime_cfg,
                                                  const ModelConfig& model_cfg_in) {
    const auto sig = dispatch_detail::signatureOf(model_cfg_in);
    if (sig.architecture.empty()) return std::nullopt;

    // SSD: declared by Genie's own `dialog.type == "ssd-q1"`. The dialog.type
    // signal is more reliable than checking model_cfg.forecast_prefix_path
    // because it doesn't require the caller to know the model is special.
    if (sig.dialog_type == "ssd-q1") {
        if (sig.architecture != "LlamaForCausalLM") {
            GENIEX_LOG_ERROR("dispatch: ssd-q1 dialog requested with unsupported architecture '{}'",
                             sig.architecture);
            return std::nullopt;
        }
        const auto cfg = dispatch_detail::autoDiscoverForecastPrefix(model_cfg_in);
        return llama3_2_3b_ssd::makePipeline(runtime_cfg, cfg);
    }

    // Standard LLM dispatch by architecture string.
    if (sig.architecture == "LlamaForCausalLM") {
        if (dispatch_detail::containsCaseInsensitive(sig.name_or_path, "falcon")) {
            return falcon3::makePipeline(runtime_cfg, model_cfg_in);
        }
        return llama3::makePipeline(runtime_cfg, model_cfg_in);
    }
    if (sig.architecture == "Qwen3ForCausalLM") return qwen3::makePipeline(runtime_cfg, model_cfg_in);
    if (sig.architecture == "Qwen2ForCausalLM") return qwen2_5::makePipeline(runtime_cfg, model_cfg_in);
    if (sig.architecture == "Phi3ForCausalLM" || sig.architecture == "Phi3VForCausalLM")
        return phi3_5::makePipeline(runtime_cfg, model_cfg_in);

    GENIEX_LOG_ERROR("dispatch: no LLM factory for architecture '{}'", sig.architecture);
    return std::nullopt;
}

// Single VLM entry point. Dispatches to the correct family factory based on
// config.json's architecture string.
inline std::optional<VLMPipeline> makeVLMPipeline(const QnnRuntimeConfig& runtime_cfg, const VLMConfig& config) {
    const auto sig = dispatch_detail::signatureOf(config.llm_config);
    if (sig.architecture.empty()) return std::nullopt;

    if (sig.architecture == "Qwen2_5_VLForConditionalGeneration")
        return qwen2_5_vl::makePipeline(runtime_cfg, config);

    GENIEX_LOG_ERROR("dispatch: no VLM factory for architecture '{}'", sig.architecture);
    return std::nullopt;
}

}  // namespace geniex
