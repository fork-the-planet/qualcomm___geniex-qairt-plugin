// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for core/src/vlm/vlm_input_provider.cpp: PrecomputedEmbeddingProvider
// (prefill-buffer + decode-lookup modes, padding) and MRoPEInputProvider (3D RoPE
// cos/sin tables, BLOCK/STRIDE interleaving, decode deltas). Each writes into a
// real Graph backed by a ClientBuffer IOTensor; no QNN runtime.

#include "vlm/vlm_input_provider.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "IOTensor.hpp"
#include "graph.h"
#include "llm/llm_types.h"
#include "testing/graph_info_builder.hpp"

namespace {

using geniex::testing::GraphInfoBuilder;

geniex::Graph makeGraph(GraphInfoBuilder& b, IOTensor& io) {
    geniex::Graph g(&b.graphInfo(), /*api=*/nullptr, &io);
    EXPECT_TRUE(g.setup(nullptr));
    return g;
}

std::string writeRawTable(const std::vector<float>& table, const std::string& tag) {
    auto          path = (std::filesystem::temp_directory_path() / ("geniex_vlm_embed_" + tag + ".bin")).string();
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(table.data()), static_cast<std::streamsize>(table.size() * sizeof(float)));
    return path;
}

// Row r of the embedding table = {r, r+0.5, r+1.0, ...}.
std::vector<float> makeTable(size_t vocab, size_t hidden) {
    std::vector<float> t(vocab * hidden);
    for (size_t r = 0; r < vocab; ++r)
        for (size_t c = 0; c < hidden; ++c) t[r * hidden + c] = static_cast<float>(r) + 0.5f * static_cast<float>(c);
    return t;
}

}  // namespace

// ─── PrecomputedEmbeddingProvider ────────────────────────────────────────────

// loadTable + lookupBatch returns the requested rows flattened.
TEST(PrecomputedEmbedding, LookupBatchReturnsRows) {
    const size_t                         vocab = 4, hidden = 3;
    geniex::PrecomputedEmbeddingProvider provider("input_embeds");
    const std::string                    path = writeRawTable(makeTable(vocab, hidden), "lookup");
    provider.loadTable(path, vocab, hidden);

    const auto out = provider.lookupBatch({2, 0});
    EXPECT_EQ(out, (std::vector<float>{2.0f, 2.5f, 3.0f, 0.0f, 0.5f, 1.0f}));
    std::remove(path.c_str());
}

// loadTable is idempotent: a second call with a different file is ignored.
TEST(PrecomputedEmbedding, LoadTableIsIdempotent) {
    const size_t                         vocab = 2, hidden = 2;
    geniex::PrecomputedEmbeddingProvider provider("input_embeds");
    const std::string                    path = writeRawTable(makeTable(vocab, hidden), "idem");
    provider.loadTable(path, vocab, hidden);
    provider.loadTable("nonexistent_path_should_be_ignored.bin", vocab, hidden);  // no throw
    EXPECT_EQ(provider.lookupBatch({1}), (std::vector<float>{1.0f, 1.5f}));
    std::remove(path.c_str());
}

// A wrong vocab/hidden vs. file size throws.
TEST(PrecomputedEmbedding, LoadTableSizeMismatchThrows) {
    geniex::PrecomputedEmbeddingProvider provider("input_embeds");
    const std::string                    path = writeRawTable(makeTable(4, 3), "mismatch");
    EXPECT_THROW(provider.loadTable(path, /*vocab=*/4, /*hidden=*/4), std::runtime_error);
    std::remove(path.c_str());
}

// Prefill mode: write() slices the per-round buffer into the graph tensor.
TEST(PrecomputedEmbedding, PrefillWritesBufferSlice) {
    const size_t     hidden = 2, rows = 2;
    GraphInfoBuilder b(
        "g", {{"input_embeds", QNN_DATATYPE_FLOAT_32, {rows, hidden}}}, {{"out", QNN_DATATYPE_FLOAT_32, {1}}});
    IOTensor      io(BufferAlloc::DEFAULT);
    geniex::Graph g = makeGraph(b, io);

    geniex::PrecomputedEmbeddingProvider provider("input_embeds");
    // Load a table so hidden_size_ is known (buffer slicing uses hidden_size_).
    const std::string path = writeRawTable(makeTable(4, hidden), "prefill");
    provider.loadTable(path, 4, hidden);
    provider.setBuffer({10.0f, 11.0f, 12.0f, 13.0f}, /*n_past_offset=*/0);

    const std::vector<int32_t>  tokens = {0, 0};
    const geniex::LLMRunContext ctx{tokens, /*n_past=*/0, /*curr_len=*/2, /*phase=*/0};
    provider.write(g, ctx);

    const auto* got = static_cast<const float*>(g.inputPtr("input_embeds"));
    EXPECT_EQ(std::vector<float>(got, got + rows * hidden), (std::vector<float>{10, 11, 12, 13}));
    std::remove(path.c_str());
}

// Decode mode (clearBuffer): write() falls back to per-token table lookup.
TEST(PrecomputedEmbedding, DecodeWritesTableLookup) {
    const size_t     hidden = 3, vocab = 4;
    GraphInfoBuilder b(
        "g", {{"input_embeds", QNN_DATATYPE_FLOAT_32, {1, hidden}}}, {{"out", QNN_DATATYPE_FLOAT_32, {1}}});
    IOTensor      io(BufferAlloc::DEFAULT);
    geniex::Graph g = makeGraph(b, io);

    geniex::PrecomputedEmbeddingProvider provider("input_embeds");
    const std::string                    path = writeRawTable(makeTable(vocab, hidden), "decode");
    provider.loadTable(path, vocab, hidden);
    provider.setBuffer({0, 0, 0}, 0);
    provider.clearBuffer();  // -> decode mode

    const std::vector<int32_t>  tokens = {3};
    const geniex::LLMRunContext ctx{tokens, /*n_past=*/5, /*curr_len=*/1, /*phase=*/1};
    provider.write(g, ctx);

    const auto* got = static_cast<const float*>(g.inputPtr("input_embeds"));
    EXPECT_EQ(std::vector<float>(got, got + hidden), (std::vector<float>{3.0f, 3.5f, 4.0f}));
    std::remove(path.c_str());
}

// Absent target tensor → silent no-op.
TEST(PrecomputedEmbedding, MissingTensorIsNoOp) {
    GraphInfoBuilder b("g", {{"other", QNN_DATATYPE_FLOAT_32, {2}}}, {{"out", QNN_DATATYPE_FLOAT_32, {2}}});
    IOTensor         io(BufferAlloc::DEFAULT);
    geniex::Graph    g = makeGraph(b, io);

    geniex::PrecomputedEmbeddingProvider provider("input_embeds");
    const geniex::LLMRunContext          ctx{{1}, 0, 1, 0};
    EXPECT_NO_THROW(provider.write(g, ctx));
}

// ─── MRoPEInputProvider ──────────────────────────────────────────────────────

// At position 0, cos = 1 and sin = 0 for every frequency (BLOCK style).
TEST(MRoPEProvider, PrefillPositionZeroIsIdentity) {
    const size_t     rows = 2, half = 4;  // mrope_section sums to 4
    GraphInfoBuilder b("g",
        {{"position_ids_cos", QNN_DATATYPE_FLOAT_32, {rows, half}},
            {"position_ids_sin", QNN_DATATYPE_FLOAT_32, {rows, half}}},
        {{"out", QNN_DATATYPE_FLOAT_32, {1}}});
    IOTensor         io(BufferAlloc::DEFAULT);
    geniex::Graph    g = makeGraph(b, io);

    geniex::MRoPEInputProvider provider({2, 1, 1}, 10000.0f, geniex::MRoPEInterleaving::BLOCK);
    // 3D positions all zero -> identity rotation.
    const std::vector<int32_t> pos(3 * rows, 0);
    provider.setPositionIds(pos, rows, /*n_past_offset=*/0);

    const geniex::LLMRunContext ctx{{0, 0}, /*n_past=*/0, /*curr_len=*/rows, /*phase=*/0};
    provider.write(g, ctx);

    const auto* cos = static_cast<const float*>(g.inputPtr("position_ids_cos"));
    const auto* sin = static_cast<const float*>(g.inputPtr("position_ids_sin"));
    for (size_t k = 0; k < rows * half; ++k) {
        EXPECT_NEAR(cos[k], 1.0f, 1e-5f);
        EXPECT_NEAR(sin[k], 0.0f, 1e-5f);
    }
}

// Decode mode (clearPositionIds): sequential positions; cos^2+sin^2==1.
TEST(MRoPEProvider, DecodeFallsBackToSequential) {
    const size_t     half = 4;
    GraphInfoBuilder b("g",
        {{"position_ids_cos", QNN_DATATYPE_FLOAT_32, {1, half}},
            {"position_ids_sin", QNN_DATATYPE_FLOAT_32, {1, half}}},
        {{"out", QNN_DATATYPE_FLOAT_32, {1}}});
    IOTensor         io(BufferAlloc::DEFAULT);
    geniex::Graph    g = makeGraph(b, io);

    geniex::MRoPEInputProvider provider({2, 1, 1}, 10000.0f, geniex::MRoPEInterleaving::STRIDE);
    provider.clearPositionIds();  // decode mode

    const geniex::LLMRunContext ctx{{9}, /*n_past=*/5, /*curr_len=*/1, /*phase=*/1};
    provider.write(g, ctx);

    const auto* cos = static_cast<const float*>(g.inputPtr("position_ids_cos"));
    const auto* sin = static_cast<const float*>(g.inputPtr("position_ids_sin"));
    for (size_t k = 0; k < half; ++k) {
        EXPECT_NEAR(cos[k] * cos[k] + sin[k] * sin[k], 1.0f, 1e-4f);
    }
}

// mrope_deltas_ accessors round-trip and reset.
TEST(MRoPEProvider, DeltaAccessors) {
    geniex::MRoPEInputProvider provider({2, 1, 1}, 10000.0f, geniex::MRoPEInterleaving::BLOCK);
    EXPECT_EQ(provider.mropeDeltas(), (std::vector<int32_t>{0, 0, 0}));
    provider.setMropeDeltas({3, 4, 5});
    EXPECT_EQ(provider.mropeDeltas(), (std::vector<int32_t>{3, 4, 5}));
    provider.resetMropeDeltas();
    EXPECT_EQ(provider.mropeDeltas(), (std::vector<int32_t>{0, 0, 0}));
}

// Absent cos/sin tensors → no-op.
TEST(MRoPEProvider, MissingTensorsIsNoOp) {
    GraphInfoBuilder b("g", {{"other", QNN_DATATYPE_FLOAT_32, {2}}}, {{"out", QNN_DATATYPE_FLOAT_32, {2}}});
    IOTensor         io(BufferAlloc::DEFAULT);
    geniex::Graph    g = makeGraph(b, io);

    geniex::MRoPEInputProvider  provider({2, 1, 1}, 10000.0f, geniex::MRoPEInterleaving::BLOCK);
    const geniex::LLMRunContext ctx{{1}, 0, 1, 0};
    EXPECT_NO_THROW(provider.write(g, ctx));
}
