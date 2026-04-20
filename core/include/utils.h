#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>

#include "geniex_export.h"

namespace geniex {

// ── IEEE 754 float16 ↔ float32 conversion ──────────────────────────────────
// Bit-level conversion; no hardware fp16 instructions required.

GENIEX_API void floatToFloat16(uint16_t* out, const float* in, size_t n);
GENIEX_API void float16ToFloat(float* out, const uint16_t* in, size_t n);

// Type alias for the timing map produced by Graph::execute().
// key   = graph/op name
// value = { cumulative_duration_us, call_count }
using TimeLog = std::map<std::string, std::pair<double, uint16_t>>;

// Returns the sum of all cumulative durations in the log, in microseconds.
GENIEX_API double totalMs(const TimeLog& log);

// Merges src into dst, accumulating durations and call counts for matching keys.
GENIEX_API void mergeTimeLogs(TimeLog& dst, const TimeLog& src);

// Prints each entry as "  name : Xus (N calls)" to stdout.
void printTimings(const TimeLog& log);

} // namespace geniex
