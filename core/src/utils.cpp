#include "utils.h"
#include "logging.h"

#include <cstring>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace geniex {

// ── IEEE 754 float16 ↔ float32 conversion ──────────────────────────────────

static uint16_t floatToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0) return static_cast<uint16_t>(sign);               // underflow → ±0
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00);     // overflow  → ±inf
    return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
}

static float halfToFloat(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        f = sign;  // ±0 (subnormals treated as zero for simplicity)
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);  // inf / NaN
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

void floatToFloat16(uint16_t* out, const float* in, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = floatToHalf(in[i]);
}

void float16ToFloat(float* out, const uint16_t* in, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = halfToFloat(in[i]);
}

double totalMs(const TimeLog& log) {
    double sum = 0.0;
    for (const auto& kv : log)
        sum += kv.second.first;
    return sum;
}

void mergeTimeLogs(TimeLog& dst, const TimeLog& src) {
    for (const auto& kv : src) {
        dst[kv.first].first  += kv.second.first;
        dst[kv.first].second += kv.second.second;
    }
}

void printTimings(const TimeLog& log) {
    std::ostringstream oss;
    for (const auto& kv : log) {
        oss << "  " << std::left << std::setw(60) << kv.first
            << std::right << std::setw(10) << std::fixed << std::setprecision(2)
            << kv.second.first << " us"
            << "  (" << kv.second.second << " calls)\n";
    }
    GENIEX_LOG_INFO("{}", oss.str());
}

} // namespace geniex
