// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// CPU unit tests for computeMRoPEPositions (core/src/vlm/vlm_utils.cpp).
// Pure state machine: assigns 3D (temporal, height, width) positions to text,
// image, and audio tokens. No QNN/OpenCV deps.

#include "vlm/vlm_utils.h"

#include <gtest/gtest.h>

using geniex::AudioSegmentInfo;
using geniex::computeMRoPEPositions;
using geniex::ImageGrid;
using geniex::MRoPEPositions;

namespace {

constexpr int32_t kVisionStart = 100;
constexpr int32_t kImageTok    = 101;
constexpr int32_t kAudioStart  = 102;
constexpr int32_t kAudioTok    = 103;

// Convenience accessors into the flat [3 * seq_len] layout.
struct Pos {
    const MRoPEPositions& p;
    size_t                seq_len;
    int32_t               t(size_t i) const { return p.position_ids[i]; }
    int32_t               h(size_t i) const { return p.position_ids[seq_len + i]; }
    int32_t               w(size_t i) const { return p.position_ids[2 * seq_len + i]; }
};

// Text-only: every token advances all three dims by one (== its index).
TEST(MRoPEPositions, TextOnlyIsSequential) {
    const std::vector<int32_t> ids = {1, 2, 3, 4};
    const auto out = computeMRoPEPositions(ids, {}, {}, 2, kVisionStart, kImageTok, kAudioStart, kAudioTok);

    ASSERT_EQ(out.position_ids.size(), 3 * ids.size());
    Pos pos{out, ids.size()};
    for (size_t i = 0; i < ids.size(); ++i) {
        EXPECT_EQ(pos.t(i), static_cast<int32_t>(i));
        EXPECT_EQ(pos.h(i), static_cast<int32_t>(i));
        EXPECT_EQ(pos.w(i), static_cast<int32_t>(i));
    }
    // No vision/audio expansion -> max position == seq_len-1 -> delta 0.
    EXPECT_EQ(out.deltas[0], 0);
    EXPECT_EQ(out.deltas[1], 0);
    EXPECT_EQ(out.deltas[2], 0);
}

// One image: vision_start is sequential, then image tokens get grid positions.
// Grid T=1,H=2,W=2 with spatial_merge_size=1 -> 4 image tokens at (h,w) in {0,1}^2.
TEST(MRoPEPositions, ImageGetsGridPositions) {
    const std::vector<int32_t>   ids   = {1, kVisionStart, kImageTok, kImageTok, kImageTok, kImageTok, 2};
    const std::vector<ImageGrid> grids = {{1, 2, 2}};
    const auto out = computeMRoPEPositions(ids, grids, {}, 1, kVisionStart, kImageTok, kAudioStart, kAudioTok);

    Pos pos{out, ids.size()};
    // idx0 text -> 0; idx1 vision_start -> 1; image grid anchored at st=2.
    EXPECT_EQ(pos.t(0), 0);
    EXPECT_EQ(pos.t(1), 1);
    // 4 image tokens: (h,w) = (0,0)(0,1)(1,0)(1,1), all temporal == img_start (2).
    EXPECT_EQ(pos.t(2), 2);
    EXPECT_EQ(pos.h(2), 2);
    EXPECT_EQ(pos.w(2), 2);
    EXPECT_EQ(pos.t(3), 2);
    EXPECT_EQ(pos.h(3), 2);
    EXPECT_EQ(pos.w(3), 3);
    EXPECT_EQ(pos.t(4), 2);
    EXPECT_EQ(pos.h(4), 3);
    EXPECT_EQ(pos.w(4), 2);
    EXPECT_EQ(pos.t(5), 2);
    EXPECT_EQ(pos.h(5), 3);
    EXPECT_EQ(pos.w(5), 3);
    // Trailing text resumes at st = img_start + max(T, llm_h, llm_w) = 2 + 2 = 4.
    EXPECT_EQ(pos.t(6), 4);
}

// One audio segment: audio tokens get sequential positions in all dims.
TEST(MRoPEPositions, AudioIsSequential) {
    const std::vector<int32_t>          ids = {kAudioStart, kAudioTok, kAudioTok, 5};
    const std::vector<AudioSegmentInfo> aud = {{/*num_llm_tokens=*/2}};
    const auto out = computeMRoPEPositions(ids, {}, aud, 1, kVisionStart, kImageTok, kAudioStart, kAudioTok);

    Pos pos{out, ids.size()};
    EXPECT_EQ(pos.t(0), 0);  // audio_start
    EXPECT_EQ(pos.t(1), 1);
    EXPECT_EQ(pos.h(1), 1);
    EXPECT_EQ(pos.w(1), 1);
    EXPECT_EQ(pos.t(2), 2);
    EXPECT_EQ(pos.h(2), 2);
    EXPECT_EQ(pos.w(2), 2);
    EXPECT_EQ(pos.t(3), 3);  // trailing text
}

// Empty input -> empty positions. delta = max_pos + 1 - seq_len = 0 + 1 - 0 = 1.
TEST(MRoPEPositions, EmptyInput) {
    const auto out = computeMRoPEPositions({}, {}, {}, 2, kVisionStart, kImageTok, kAudioStart, kAudioTok);
    EXPECT_TRUE(out.position_ids.empty());
    EXPECT_EQ(out.deltas[0], 1);
}

// vision_start with no matching grid falls through to sequential text handling.
TEST(MRoPEPositions, VisionStartWithoutGridIsSequential) {
    const std::vector<int32_t> ids = {kVisionStart, 1, 2};
    const auto                 out =
        computeMRoPEPositions(ids, /*no grids*/ {}, {}, 1, kVisionStart, kImageTok, kAudioStart, kAudioTok);
    Pos pos{out, ids.size()};
    EXPECT_EQ(pos.t(0), 0);
    EXPECT_EQ(pos.t(1), 1);
    EXPECT_EQ(pos.t(2), 2);
}

}  // namespace
