---
name: write-xtensor
description: Write C++ tensor logic using xtensor (NumPy-like API) for prototyping before QNN conversion
allowed-tools: Read, Edit, Write, Bash
---

# Write xtensor code

Use this skill when prototyping tensor manipulation logic. xtensor provides NumPy-like C++ syntax for rapid development. The resulting code can later be converted to direct QNN buffer operations using `/convert-xtensor-to-qnn`.

## Core reference

| Operation | PyTorch/NumPy | xtensor |
|-----------|---------------|---------|
| Zeros | `np.zeros((3,4))` | `xt::zeros<double>({3, 4})` |
| Ones | `np.ones((3,4))` | `xt::ones<double>({3, 4})` |
| From data | `np.array([[1,2],[3,4]])` | `xt::xarray<int>{{1,2},{3,4}}` |
| Range | `np.arange(0,10,2)` | `xt::arange<int>(0, 10, 2)` |
| Linspace | `np.linspace(0,1,5)` | `xt::linspace<double>(0, 1, 5)` |
| Element access | `a[0,1]` | `a(0, 1)` |
| Slice | `a[1:3, :]` | `xt::view(a, xt::range(1,3), xt::all())` |
| Boolean index | `a[a > 0]` | `xt::filter(a, a > 0)` |
| Matmul | `np.dot(a, b)` | `xt::linalg::dot(a, b)` |
| Reshape | `a.reshape(2,3)` | `xt::reshape_view(a, {2, 3})` |
| Transpose | `a.T` | `xt::transpose(a)` |
| Flatten | `a.flatten()` | `xt::flatten(a)` |
| Concat | `np.concatenate([a,b], 0)` | `xt::concatenate(xt::xtuple(a, b), 0)` |
| Stack | `np.stack([a,b])` | `xt::stack(xt::xtuple(a, b))` |
| Cast | `a.astype(float)` | `xt::cast<double>(a)` |
| Sum | `np.sum(a)` | `xt::sum(a)` |
| Mean | `np.mean(a)` | `xt::mean(a)` |
| Max/Min | `np.max(a)` / `np.min(a)` | `xt::amax(a)` / `xt::amin(a)` |
| Argmax | `np.argmax(a)` | `xt::argmax(a)` |
| Row select | `a[indices]` | `xt::view(a, xt::keep(indices), xt::all())` |

## std::vector conversions

```cpp
// vector → xarray
xt::adapt(vec)                          // 1D (shape inferred)
xt::adapt(vec, shape)                   // explicit shape

// xarray → vector
std::vector<T> vec(arr.begin(), arr.end());
```

## Critical pitfalls

1. **`xt::split` — never use `auto` for element access**:
   ```cpp
   auto chunks = xt::split(arr, 2, 0);
   xt::xarray<float> chunk0 = chunks[0];  // explicit type required
   ```

2. **`xt::concatenate` — never self-assign**:
   ```cpp
   a = xt::concatenate(xt::xtuple(a, b), 0);                   // WRONG
   xt::xarray<float> c = xt::concatenate(xt::xtuple(a, b), 0); // correct
   ```

3. **`xt::filter` — use `xt::equal` for comparisons**:
   ```cpp
   xt::filter(a, xt::equal(a, b)) += 1;  // correct
   xt::filter(a, a == b) += 1;           // WRONG
   ```

4. **`xt::interp` — requires monotonically increasing x**. Flip descending data first, interpolate, then flip back.

5. **Lazy evaluation** — xtensor is lazy by default. Use `xt::eval(expr)` to force materialization when needed.

6. **Shape type** — uses `std::vector<size_t>` for shape arguments.

7. **Print shape** — use `xt::adapt(arr.shape())` not `arr.shape()` directly.

## Includes

```cpp
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>
// For linalg:
#include <xtensor-blas/xlinalg.hpp>
```