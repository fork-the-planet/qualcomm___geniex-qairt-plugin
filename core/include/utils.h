// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>

#include "geniex_export.h"

namespace geniex {

// Bit-level float16 ↔ float32 conversion; no hardware fp16 instructions required.
GENIEX_API void floatToFloat16(uint16_t* out, const float* in, size_t n);
GENIEX_API void float16ToFloat(float* out, const uint16_t* in, size_t n);

// Timing map produced by Graph::execute().
// key = graph/op name, value = { cumulative_duration_us, call_count }
using TimeLog = std::map<std::string, std::pair<double, uint16_t>>;

GENIEX_API double totalMs(const TimeLog& log);
GENIEX_API void mergeTimeLogs(TimeLog& dst, const TimeLog& src);

void printTimings(const TimeLog& log);

} // namespace geniex
