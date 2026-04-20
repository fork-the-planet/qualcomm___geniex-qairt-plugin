# xtensor to QNN tensor Conversion

A guide for converting xtensor tensors to QNN tensor format.

---

## Overview

QNN is a native library and tensor format for the Qualcomm QNN backend. This document covers the conversion process from xtensor to QNN tensors, with focus on memory management and buffer handling.

---

## Shared Buffers

### Concept

Shared buffers allow multiple graphs to share not only weights but also input and output tensors. There are two allocation strategies:

| Strategy | Description | Trade-off |
|----------|-------------|-----------|
| **Separate buffers** | Allocate individual memory for each input | Higher memory usage |
| **Shared buffer** | Allocate one large buffer (maximum required size) | Memory efficient |

For LLM inference, since ARN and AR1 graphs do not run in an interleaved manner, a single shared buffer (provided by the NPU SDK) is sufficient.

### Memory Layout

QNN uses **row-major** (C-style) memory layout, same as the C default.

For a 2D tensor:
```
X[0,0] = a,  X[0,1] = b,  X[1,0] = c,  X[1,1] = d
Memory layout: [a, b, c, d]
```

### Buffer Sharing Behavior

When passing values between graphs (e.g., ARN output → AR1 input), ensure proper memory layout adjustment. With shared buffers, modifying one input affects the other since they share the same underlying memory.

---

## Reshape Examples

### Expand: A (2×2) → B (2×3)

| Step | Memory Layout | Notes |
|------|---------------|-------|
| Start | `[a, b, c, d, ?, ?]` | Original data |
| After | `[a, b, 0, c, d, 0]` | Zero-padded |

- Buffer is extended with zero-padding
- Process **last row to first** (moving values backward)

### Shrink: B (2×3) → A (2×2)

| Step | Memory Layout | Notes |
|------|---------------|-------|
| Start | `[a, b, 0, c, d, 0]` | Original data |
| After | `[a, b, c, d, 0, 0]` | Trailing values ignored |

- Padding is removed and memory compacted
- Process **first row to last** (moving values forward)