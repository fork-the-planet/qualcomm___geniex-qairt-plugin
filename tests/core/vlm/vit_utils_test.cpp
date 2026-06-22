// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// CPU unit tests for the Qwen-ViT pure-math helpers (core/src/vlm/vit_utils.cpp).
// All functions are deterministic standard-library math with no QNN/OpenCV deps.

#include "vlm/vit_utils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <set>

using namespace geniex::qwen_vit;

namespace {

// inv_freq[i] = 1 / theta^(2i/dim).
TEST(MakeInvFreq, MatchesClosedForm) {
    const int   dim   = 8;
    const float theta = 10000.0f;
    const auto  inv   = makeInvFreq(dim, theta);

    ASSERT_EQ(inv.size(), static_cast<size_t>(dim / 2));
    EXPECT_FLOAT_EQ(inv[0], 1.0f);  // i=0 -> theta^0 = 1
    for (int i = 0; i < dim / 2; ++i) {
        EXPECT_FLOAT_EQ(inv[i], 1.0f / std::pow(theta, float(i * 2) / float(dim)));
    }
    // Frequencies strictly decrease.
    for (size_t i = 1; i < inv.size(); ++i) EXPECT_LT(inv[i], inv[i - 1]);
}

// With no merge/window padding, the window index is a permutation of [0, n_tok).
TEST(ComputeWindowIndex, IsPermutationOfTokens) {
    const int  T = 1, H = 8, W = 8;
    const int  spatial_merge_size = 2, window_size = 8, patch_size = 2;
    const auto idx = computeWindowIndex(T, H, W, spatial_merge_size, window_size, patch_size);

    const int    llm_h = H / spatial_merge_size, llm_w = W / spatial_merge_size;
    const size_t n_tok = static_cast<size_t>(T * llm_h * llm_w);
    ASSERT_EQ(idx.size(), n_tok);

    std::set<int64_t> seen(idx.begin(), idx.end());
    EXPECT_EQ(seen.size(), n_tok);                               // no duplicates
    EXPECT_EQ(*seen.begin(), 0);                                 // min index
    EXPECT_EQ(*seen.rbegin(), static_cast<int64_t>(n_tok - 1));  // max index
}

// reverseWindowIndex composed with the forward index is the identity.
TEST(ReverseWindowIndex, InvertsForwardPermutation) {
    const auto fwd = computeWindowIndex(1, 8, 8, 2, 8, 2);
    const auto rev = reverseWindowIndex(fwd);
    ASSERT_EQ(rev.size(), fwd.size());
    for (size_t i = 0; i < fwd.size(); ++i) {
        EXPECT_EQ(rev[static_cast<size_t>(fwd[i])], static_cast<int64_t>(i));
    }
}

// computeSpatialCosSin: shape, and cos^2 + sin^2 == 1 for every element.
TEST(ComputeSpatialCosSin, ShapeAndPythagorean) {
    const int  T = 1, H = 8, W = 8, sm = 2;
    const auto inv        = makeInvFreq(8, 10000.0f);
    const auto idx        = computeWindowIndex(T, H, W, sm, 8, 2);
    const auto [cos, sin] = computeSpatialCosSin(T, H, W, sm, inv, idx);

    const int    llm_h = H / sm, llm_w = W / sm, sm_unit = sm * sm;
    const int    emb_dim = static_cast<int>(inv.size()) * 2;
    const size_t seq_len = static_cast<size_t>(T * llm_h * llm_w * sm_unit);
    ASSERT_EQ(cos.size(), seq_len * emb_dim);
    ASSERT_EQ(sin.size(), seq_len * emb_dim);

    for (size_t k = 0; k < cos.size(); ++k) {
        EXPECT_NEAR(cos[k] * cos[k] + sin[k] * sin[k], 1.0f, 1e-5f);
    }
}

// windowReorder moves whole (sm_unit x embed_dim) groups per the window index.
TEST(WindowReorder, MovesGroupsByIndex) {
    const size_t               n_groups = 3, sm_unit = 1, embed_dim = 2;
    const std::vector<float>   hidden       = {0, 0, 1, 1, 2, 2};  // group g = {g, g}
    const std::vector<int64_t> window_index = {2, 0, 1};           // out[0]=grp2, out[1]=grp0, out[2]=grp1
    const auto                 out          = windowReorder(hidden, n_groups, sm_unit, embed_dim, window_index);

    ASSERT_EQ(out.size(), hidden.size());
    EXPECT_FLOAT_EQ(out[0], 2);
    EXPECT_FLOAT_EQ(out[1], 2);
    EXPECT_FLOAT_EQ(out[2], 0);
    EXPECT_FLOAT_EQ(out[3], 0);
    EXPECT_FLOAT_EQ(out[4], 1);
    EXPECT_FLOAT_EQ(out[5], 1);
}

// computePatchRoPE: shape [T*H*W, 2*half] and unit-circle invariant.
TEST(ComputePatchRoPE, ShapeAndPythagorean) {
    const int  T = 1, H = 4, W = 4, sm = 2;
    const auto inv        = makeInvFreq(8, 10000.0f);
    const auto [cos, sin] = computePatchRoPE(T, H, W, sm, inv);

    const size_t n     = static_cast<size_t>(T * H * W);
    const int    embed = static_cast<int>(inv.size()) * 2;
    ASSERT_EQ(cos.size(), n * embed);
    ASSERT_EQ(sin.size(), n * embed);
    for (size_t k = 0; k < cos.size(); ++k) {
        EXPECT_NEAR(cos[k] * cos[k] + sin[k] * sin[k], 1.0f, 1e-5f);
    }
}

// computeCuWindowSeqlens: monotonic non-decreasing prefix sums starting at 0.
TEST(ComputeCuWindowSeqlens, MonotonicPrefixSums) {
    const auto cu = computeCuWindowSeqlens(1, 8, 8, 2, 8, 2);
    ASSERT_FALSE(cu.empty());
    EXPECT_EQ(cu.front(), 0);
    for (size_t i = 1; i < cu.size(); ++i) EXPECT_GE(cu[i], cu[i - 1]);
}

// buildBlockAttentionMask: 0 within a segment, blocked across segments.
TEST(BuildBlockAttentionMask, BlockDiagonal) {
    const size_t               N          = 4;
    const std::vector<int64_t> boundaries = {0, 2, 4};  // two 2x2 blocks
    const auto                 mask       = buildBlockAttentionMask(N, boundaries, /*allowed=*/0.0f, /*blocked=*/-1e9f);
    ASSERT_EQ(mask.size(), N * N);

    auto at = [&](size_t r, size_t c) { return mask[r * N + c]; };
    // Same block (0,1) and (2,3) -> allowed.
    EXPECT_FLOAT_EQ(at(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(at(3, 2), 0.0f);
    // Cross block -> blocked.
    EXPECT_FLOAT_EQ(at(0, 2), -1e9f);
    EXPECT_FLOAT_EQ(at(3, 0), -1e9f);
}

}  // namespace
