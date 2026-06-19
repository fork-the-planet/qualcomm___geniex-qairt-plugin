// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Test-side controls for the link-time QnnApi stub (testing/stub_qnnapi.cpp).

#pragma once

#include <cstdint>

namespace geniex::testing {

// When >= 0, the stub's graphExecute writes a one-hot peak at this token id
// into any output tensor named "logits" (every row), instead of the default
// identity copy. -1 restores pure identity-copy behaviour.
void stubSetNextToken(int32_t token_id);

// Vocabulary size used to interpret each logits row; must match the fixture's
// logits tensor inner dimension. Defaults to 0 (no logits write).
void stubSetVocabSize(uint32_t vocab_size);

}  // namespace geniex::testing
