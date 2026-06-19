// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for core/src/llm/llm_model.cpp - the prefill/decode orchestration
// loop. Drives a real LLMModel against a CPU-only graph fixture and the
// link-time QnnApi stub; no QNN device bring-up. A test-only subclass injects
// the loaded graphs and calls onInitialized() to bypass Model::initialize().

#include "llm/llm_model.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "QnnApi.hpp"
#include "testing/llm_fixture.hpp"
#include "testing/stub_qnnapi.hpp"

namespace {

using geniex::testing::LLMFixture;

// Exposes the protected Model members so a test can wire in pre-built graphs
// and run the orchestration entry points without a QNN backend.
class TestableLLMModel : public geniex::LLMModel {
   public:
    explicit TestableLLMModel(geniex::LLMSpec spec) : geniex::LLMModel(std::move(spec)) {}

    // Moves the fixture's graphs into the model and runs onInitialized(). The
    // fixture (which owns the IOTensor / QnnApi / tensor buffers the graphs
    // point at) must outlive this model.
    bool initFromFixture(LLMFixture& fx) {
        api_       = std::make_unique<QnnApi>();
        io_tensor_ = std::shared_ptr<IOTensor>(std::shared_ptr<void>{}, &fx.io);  // non-owning alias
        for (auto& g : fx.graphs) graphs_.push_back(std::move(g));
        const bool ok = onInitialized();
        initialized_  = ok;
        return ok;
    }
};

// Decode runs serially when no worker pool is created; the pool is only built
// when GENIEX_DECODE_WORKERS or the clock keeper is enabled. Force both off so
// updateKV happens inline and the loop is fully deterministic.
struct NoDecodePoolEnv {
    NoDecodePoolEnv() {
        _putenv_s("GENIEX_DECODE_WORKERS", "0");
        _putenv_s("GENIEX_CLOCK_KEEPER_THREADS", "0");
    }
    ~NoDecodePoolEnv() {
        _putenv_s("GENIEX_DECODE_WORKERS", "");
        _putenv_s("GENIEX_CLOCK_KEEPER_THREADS", "");
    }
};

// Builds an initialized model over a fresh fixture. Holds both alive.
struct ModelFixture {
    NoDecodePoolEnv  no_pool;
    LLMFixture       fx;
    TestableLLMModel model{LLMFixture::makeSpec()};

    ModelFixture() { EXPECT_TRUE(model.initFromFixture(fx)); }
};

geniex::GenerationConfig greedyConfig(int32_t max_tokens) {
    geniex::GenerationConfig cfg;
    cfg.enable_sampling = false;  // argmax fast path
    cfg.max_tokens      = max_tokens;
    return cfg;
}

}  // namespace

// onInitialized derives the runtime shape from the loaded graph names.
TEST(LLMModel, InitializesFromGraphNames) {
    ModelFixture mf;
    EXPECT_EQ(mf.model.nPast(), 0u);
}

// A short prefill + single decode step emits exactly the token the stub was
// told to produce (greedy argmax, sampling disabled).
TEST(LLMModel, GreedyDecodeEmitsStubToken) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(5);

    const std::vector<int32_t> prompt = {1, 2, 3};
    auto                       out    = mf.model.generate(prompt, greedyConfig(/*max_tokens=*/1));

    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0], 5);
    EXPECT_EQ(mf.model.nPast(), prompt.size() + 1);

    geniex::testing::stubSetNextToken(-1);
}

// Generation stops at an EOS token and excludes it from the output.
TEST(LLMModel, StopsOnEosAndExcludesIt) {
    geniex::LLMSpec spec = LLMFixture::makeSpec();
    spec.eos_token_ids   = {7};

    NoDecodePoolEnv  no_pool;
    LLMFixture       fx;
    TestableLLMModel model{spec};
    ASSERT_TRUE(model.initFromFixture(fx));

    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(7);  // first sampled token is EOS

    auto out = model.generate({1, 2}, greedyConfig(/*max_tokens=*/5));
    EXPECT_TRUE(out.empty());

    geniex::testing::stubSetNextToken(-1);
}

// token_callback returning false stops generation early.
TEST(LLMModel, CallbackStopsEarly) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(3);

    int  calls          = 0;
    auto stop_after_two = [&calls](int32_t) { return ++calls < 2; };
    auto out            = mf.model.generate({1}, greedyConfig(/*max_tokens=*/10), stop_after_two);

    EXPECT_EQ(calls, 2);
    EXPECT_EQ(out.size(), 2u);

    geniex::testing::stubSetNextToken(-1);
}

// A prompt longer than the max context length is rejected up front.
TEST(LLMModel, ThrowsWhenPromptExceedsContext) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(0);

    std::vector<int32_t> prompt(LLMFixture::kContextLen + 1, 1);
    EXPECT_THROW(mf.model.generate(prompt, greedyConfig(/*max_tokens=*/1)), geniex::ContextLengthExceededError);

    geniex::testing::stubSetNextToken(-1);
}
