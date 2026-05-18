---
name: convert-xtensor-to-qnn
description: Convert xtensor prototype code to production QNN direct-buffer operations (zero-copy)
allowed-tools: Read, Edit, Write, Bash
---

# Convert xtensor to QNN tensors

Convert xtensor-based prototype code to direct QNN buffer operations for production. This eliminates the extra memory copy that xtensor introduces.

## Conversion patterns

| xtensor | QNN equivalent |
|---------|----------------|
| `xt::xarray<T>` | Raw pointer + shape metadata |
| `xt::view(arr, range)` | Offset calculation into shared buffer |
| `xt::zeros<T>({...})` | `memset(ptr, 0, size)` |
| `xt::ones<T>({...})` | Loop initialization |
| `xt::adapt(ptr, shape)` | Direct pointer use (already zero-copy) |
| `xt::eval(expr)` | Materialize into pre-allocated buffer |

## QNN buffer access

```cpp
// Read from graph output
float* output_ptr = static_cast<float*>(graph.outputPtr("tensor_name"));

// Write to graph input
float* input_ptr = static_cast<float*>(graph.inputPtr("tensor_name"));

// Using Graph API (handles dtype conversion: float32, float16, ufixed8/16, int32)
graph.write("tensor_name", float_data_ptr);
graph.read("tensor_name", float_output_ptr);
```

## Memory layout

QNN uses **row-major** (C-style) layout, same as xtensor default.

For 2D tensor `X[rows][cols]`:
```
X[0,0], X[0,1], ..., X[0,cols-1], X[1,0], X[1,1], ...
```

## Shared buffers

ARN (prefill) and AR1 (decode) graphs share input/output buffers since they never run concurrently. Modifying one graph's input buffer affects the other.

## Reshape operations

**Expand (2x2 → 2x3)**: Process last row to first (move backward), zero-pad new positions.

**Shrink (2x3 → 2x2)**: Process first row to last (move forward), compact memory.

## When to convert

| Scenario | Use |
|----------|-----|
| Initial prototyping | xtensor |
| Debugging complex logic | xtensor |
| Production deployment | QNN tensors |
| Performance-critical paths | QNN tensors |
| One-off preprocessing | xtensor (acceptable overhead) |

## Workflow

1. Read the xtensor code to understand the tensor operations
2. Identify buffer pointers from `graph.inputPtr()` / `graph.outputPtr()`
3. Replace xtensor operations with direct pointer arithmetic
4. Ensure correct stride calculations for multi-dimensional access
5. Use `memset` for zero-initialization, `memcpy` for block copies
6. Verify correctness by comparing output values against the xtensor version