#pragma once

#include "graph.h"
#include "types.h"
#include "llm/llm_types.h"
#include "llm/llm_utils.h"
#include "geniex_export.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace geniex {

// Abstract interface for CPU-side tensor writes at inference time.
// Implementations write one or more named inputs into a Graph.
class GENIEX_API InputProvider {
public:
    virtual ~InputProvider() = default;

    // Called once after all Graph objects are ready.
    // Implementations that need runtime configuration (e.g. embedding path from
    // ModelConfig) perform their one-time setup here.
    virtual void onInitialized(const ModelConfig&) {}

    // Write input tensor(s) into g for the given context.
    // Implementations must silently do nothing if the target tensor is absent
    // on this shard (use Graph::hasInput() to check).
    virtual void write(Graph& g, const LLMRunContext& ctx) = 0;
};

// Owns a token embedding table and writes the [curr_len * hidden_size] embeds
// tensor for each forward pass.
class GENIEX_API EmbeddingInputProvider : public InputProvider {
public:
    // tensor_name: name of the graph input to write (default "input_embeds").
    explicit EmbeddingInputProvider(std::string tensor_name = "input_embeds");

    // Pre-loads the embedding table from a row-major float32 binary file.
    // If called before onInitialized(), the onInitialized() load is skipped.
    bool loadTable(const std::string& path, size_t vocab_size, size_t hidden_size);

    // If the table has not already been loaded via loadTable(), loads it from
    // model_cfg.embedding_path (npy format). No-op if embedding_path is empty.
    void onInitialized(const ModelConfig& model_cfg) override;

    void write(Graph& g, const LLMRunContext& ctx) override;

private:
    std::string        tensor_name_;
    std::vector<float> table_;       // flat row-major [vocab_size * hidden_size]
    size_t             hidden_size_ = 0;
};

// Writes raw token IDs (int32) into a named graph input, padding to the
// graph's tensor capacity with a configurable pad token.
// Used for models where the embedding lookup runs on-device (e.g. AI Hub exports).
class GENIEX_API TokenIdInputProvider : public InputProvider {
public:
    explicit TokenIdInputProvider(std::string tensor_name = "input_ids",
                                  int32_t pad_token_id = 0);

    void write(Graph& g, const LLMRunContext& ctx) override;

private:
    std::string tensor_name_;
    int32_t     pad_token_id_;
};

// Computes RoPE cos/sin tables and writes them into the two named graph inputs.
class GENIEX_API RoPEInputProvider : public InputProvider {
public:
    // head_dim: per-head dimension (cos/sin vectors have size curr_len * head_dim/2).
    // theta:    RoPE base frequency.
    // cos_name / sin_name: names of the two graph inputs to write.
    RoPEInputProvider(size_t head_dim, float theta,
                      std::string cos_name = "position_ids_cos",
                      std::string sin_name = "position_ids_sin");

    void write(Graph& g, const LLMRunContext& ctx) override;

private:
    RotaryEmbedding rope_;
    std::string     cos_name_;
    std::string     sin_name_;
};

// LongRoPE with dynamic scaling and per-dimension extension factors.
class GENIEX_API LongRoPEInputProvider : public InputProvider {
public:
    LongRoPEInputProvider(size_t head_dim, float theta,
                          std::vector<float> ext_factors,
                          int max_position_embeddings = 131072,
                          int original_max_position_embeddings = 4096,
                          std::string cos_name = "position_ids_cos",
                          std::string sin_name = "position_ids_sin");

    void write(Graph& g, const LLMRunContext& ctx) override;

private:
    LongRoPEEmbedding rope_;
    std::string       cos_name_;
    std::string       sin_name_;
};

// Partial RoPE (rope_fraction of head_dim) with post-scale factor.
class GENIEX_API PartialRoPEInputProvider : public InputProvider {
public:
    PartialRoPEInputProvider(size_t head_dim, float theta = 10000.f,
                             float rope_fraction = 1.0f, float scale = 1.0f,
                             std::string cos_name = "position_ids_cos",
                             std::string sin_name = "position_ids_sin");

    void write(Graph& g, const LLMRunContext& ctx) override;

private:
    PartialRoPEEmbedding rope_;
    std::string          cos_name_;
    std::string          sin_name_;
};

} // namespace geniex
