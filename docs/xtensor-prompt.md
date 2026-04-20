# xtensor Reference Guide

A reference for converting PyTorch/NumPy code to xtensor.

---

## Overview

[xtensor](https://github.com/xtensor-stack/xtensor) is a C++ library for numerical analysis with multi-dimensional array expressions, similar to NumPy.

---

## Basic Operations

The table below compares common operations between PyTorch/NumPy and xtensor:

| Operation | PyTorch | NumPy | xtensor | Notes |
|-----------|---------|-------|---------|-------|
| **Array Creation** |
| Create zeros | `torch.zeros(3, 4)` | `np.zeros((3, 4))` | `xt::zeros<double>({3, 4})` | Specify data type explicitly |
| Create ones | `torch.ones(3, 4)` | `np.ones((3, 4))` | `xt::ones<double>({3, 4})` |  |
| Create from data | `torch.tensor([[1, 2], [3, 4]])` | `np.array([[1, 2], [3, 4]])` | `xt::xarray<int>{{1, 2}, {3, 4}}` |  |
| Create range | `torch.arange(0, 10, 2)` | `np.arange(0, 10, 2)` | `xt::arange<int>(0, 10, 2)` |  |
| Create linspace | `torch.linspace(0, 1, 5)` | `np.linspace(0, 1, 5)` | `xt::linspace<double>(0, 1, 5)` |  |
| **Indexing & Slicing** |
| Element access | `tensor[0, 1]` | `array[0, 1]` | `arr(0, 1)` | Use parentheses, not brackets |
| Slice | `tensor[1:3, :]` | `array[1:3, :]` | `xt::view(arr, xt::range(1, 3), xt::all())` | Use `xt::view` with `xt::range` |
| Boolean indexing | `tensor[tensor > 0]` | `array[array > 0]` | `xt::filter(arr, arr > 0)` |  |
| **Mathematical Operations** |
| Element-wise add | `a + b` | `a + b` | `a + b` |  |
| Element-wise multiply | `a * b` | `a * b` | `a * b` |  |
| Matrix multiplication | `torch.matmul(a, b)` | `np.dot(a, b)` | `xt::linalg::dot(a, b)` | Needs `#include <xtensor-blas/xlinalg.hpp>` |
| Power | `torch.pow(a, 2)` | `np.power(a, 2)` | `xt::pow(a, 2)` |  |
| Square root | `torch.sqrt(a)` | `np.sqrt(a)` | `xt::sqrt(a)` |  |
| Exponential | `torch.exp(a)` | `np.exp(a)` | `xt::exp(a)` |  |
| **Reductions** |
| Sum | `torch.sum(a)` | `np.sum(a)` | `xt::sum(a)` |  |
| Mean | `torch.mean(a)` | `np.mean(a)` | `xt::mean(a)` |  |
| Max | `torch.max(a)` | `np.max(a)` | `xt::amax(a)` | Note: amax, not max |
| Min | `torch.min(a)` | `np.min(a)` | `xt::amin(a)` | Note: amin, not min |
| Argmax | `torch.argmax(a)` | `np.argmax(a)` | `xt::argmax(a)` |  |
| **Shape Manipulation** |
| Reshape | `tensor.reshape(2, 3)` | `array.reshape(2, 3)` | `xt::reshape_view(arr, {2, 3})` | Use reshape_view for efficient reshaping |
|  |  |  | `arr.reshape({2, 3})` | In-place reshaping |
| Transpose | `tensor.T` | `array.T` | `xt::transpose(arr)` |  |
| Flatten | `tensor.flatten()` | `array.flatten()` | `xt::flatten(arr)` |  |
| Squeeze | `torch.squeeze(tensor)` | `np.squeeze(array)` | `xt::squeeze(arr)` |  |
| **Concatenation** |
| Concatenate | `torch.cat([a, b], dim=0)` | `np.concatenate([a, b], axis=0)` | `xt::concatenate(xt::xtuple(a, b), 0)` | Use xtuple for multiple arrays |
| Stack | `torch.stack([a, b])` | `np.stack([a, b])` | `xt::stack(xt::xtuple(a, b))` | Creates new dimension |
| **Type Conversion** |
| Cast to float | `tensor.float()` | `array.astype(float)` | `xt::cast<double>(arr)` | Explicit template parameter |
| Cast to int | `tensor.int()` | `array.astype(int)` | `xt::cast<int>(arr)` | Type safety enforced |
| **Broadcasting** |
| Add scalar | `tensor + 5` | `array + 5` | `arr + 5` | Automatic broadcasting |
| Add different shapes | `a + b` | `a + b` | `a + b` | NumPy-style broadcasting rules |

---

## Conversion: `std::vector<T>` ↔ `xt::xarray<T>`

### From `std::vector<T>` to `xt::xarray<T>`

```cpp
xt::adapt(vec)                          // 1D array (shape inferred)
xt::adapt(vec, shape)                   // With explicit shape
xt::adapt(vec.begin(), vec.end())       // From iterators
xt::adapt(vec.begin(), vec.end(), shape)
```

### From `xt::xarray<T>` to `std::vector<T>`

```cpp
std::vector<T> vec(arr.begin(), arr.end());
std::vector<T> vec(arr.data(), arr.data() + arr.size());
```

---

## Important Notes

| Topic | Description |
|-------|-------------|
| **Namespace** | Most functions are in the `xt::` namespace |
| **Lazy Evaluation** | xtensor uses lazy evaluation by default. Use `xt::eval()` to force evaluation |
| **Memory Layout** | Supports both row-major (C-style) and column-major (Fortran-style) layouts |
| **Type Safety** | C++ requires explicit type specification in many cases |
| **Views vs Copies** | Many operations return views for efficiency. Use `xt::eval()` for a copy |
| **Shape** | Uses `std::vector<size_t>` for shape arguments (e.g., `xt::adapt`, `xt::reshape_view`) |

> **Tip:** To print a shape, use `std::cout << xt::adapt(arr.shape()) << std::endl;` instead of `std::cout << arr.shape() << std::endl;`.

---

## Special Cases

### `xt::interp` — Monotonic x-coordinates required

`xt::interp` requires x-coordinates to be monotonically increasing. For descending data, flip before interpolation and flip back after.

```cpp
auto timesteps = xt::arange(100, 0);
// NumPy: sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)

xt::xarray<float> interp_timesteps = xt::arange(static_cast<size_t>(0), sigmas.size());
xt::xarray<float> timesteps_i = xt::flip(timesteps, 0);  // Flip to ascending order
xt::xarray<float> sigmas_i = xt::interp(timesteps_i, interp_timesteps, sigmas);
sigmas = xt::flip(sigmas_i, 0);  // Flip back to original order
```

### `xt::split` — Avoid `auto` for element access

`xt::split` returns `std::vector<xt::xarray<T>>`. Using `auto` when accessing elements can cause memory issues and incorrect values.

```cpp
auto noise_pred_chunks = xt::split(noise_pred, 2, 0);
assert(noise_pred_chunks.size() == 2 && "CFG requires splitting noise_pred to 2 chunks");

// Explicitly specify type — do NOT use auto here
xt::xarray<float> noise_pred_uncond = noise_pred_chunks[0];
xt::xarray<float> noise_pred_text = noise_pred_chunks[1];
```

### `xt::concatenate` — Avoid self-assignment

Assigning the result back to an input array can produce incorrect results, especially with large arrays.

```cpp
xt::xarray<float> a, b = ...;
a = xt::concatenate(xt::xtuple(a, b), 0);                   // ❌ May produce wrong results
xt::xarray<float> c = xt::concatenate(xt::xtuple(a, b), 0); // ✅ Use a new array
```

### `xt::filter` — Use `xt::equal` for comparisons

When using `xt::filter` with element-wise comparison, use `xt::equal` instead of `==`.

```cpp
xt::xarray<int32_t> a = ...;
int32_t b = 5;
xt::filter(a, xt::equal(a, b)) += 1;  // ✅ Correct
xt::filter(a, a == b) += 1;           // ❌ Incorrect
```

### Row selection with `xt::keep`

To select specific rows by index:

```cpp
xt::xarray<float> inputs_embeds = xt::view(embedded_tokens, xt::keep(input_ids), xt::all());
```