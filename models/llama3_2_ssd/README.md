# Llama 3.2 3B Instruct — Self-Speculative Decoding (SSD)

## Algorithm

This model uses **EAGLE**-based tree-structured speculative decoding to produce
1–3 tokens per iteration instead of 1, using a single model and a single forward
pass per iteration.

> Li et al., *"EAGLE: Speculative Sampling Requires Rethinking Feature
> Uncertainty"*, 2024.
> <https://arxiv.org/abs/2401.15077>

### NPU adaptation: forecast tokens

EAGLE uses a lightweight draft head network to produce look-ahead logits. On NPU
with static-graph execution, a separate draft head is impractical. Instead, the
model is trained with **forecast tokens** — special token IDs beyond the vocabulary
(128256, 128257) — that produce look-ahead logits when processed by the full model.
Each of the 10 draft nodes gets a chain of 2 forecast tokens (one per tree level)
attached via the attention mask. This fuses tree verification and next-tree
preparation into one forward pass: 10 draft nodes + 20 forecast tokens = 30 tokens,
padded to AR-32.

### Forecast prefix

16 pre-computed KV cache entries loaded from disk at init to solve the cold-start
problem (no previous forecast logits before the first decode step). Prompt tokens
skip these via attention mask; forecast tokens attend to them.

## Worked example

Config: `branches=[3, 2]` → 2-level tree, 10 draft nodes, 20 forecast tokens.

Last accepted token from previous iteration: **C** (argmax output, never processed
as input). Previous forecast logits gave candidates: level-1 {D, E, F}, level-2
for D {G, H}, for E {I, J}, for F {K, L}.

**Input (32 slots, one AR-32 forward pass):**

```
 0: C (root)     1: D  2: E  3: F        ← 10 draft nodes
 4: G  5: H  6: I  7: J  8: K  9: L
10–29: forecast tokens (128256/128257, 2 per node)
30–31: padding
```

**Output (32 logit rows):**

```
 0: "after C" → X     Always accepted. Check: does D, E, or F == X?
 1: "after D" → Y₁    Only read if D == X
 2: "after E" → Y₂    Only read if E == X
 3: "after F" → Y₃    Only read if F == X
 4–9: level-2 logits   Only read if parent matched
10–29: forecast logits  Only 2 read (for last accepted node) → next tree
```

**Verification (say X = E):**

1. Index 0: argmax = E → accepted. E == E? Yes → walk into index 2.
2. Index 2: argmax = Y₂ → accepted. J == Y₂? Yes → walk into index 7.
3. Index 7: argmax = Z₄ → accepted. No more levels.

**Result:** 3 tokens emitted {E, Y₂, Z₄}. KV committed for input indices {0, 2,
7} (tokens C, E, J — already processed). Z₄ becomes the next root (never
processed, needs KV). Next tree built from forecast logits at `10 + 7×2 = 24`.

## Implementation

SSD logic is fully contained in this directory (not part of `geniex_core`). It
subclasses `LLMModel` to override the decode loop.

| File | Description |
|---|---|
| `ssd_types.h` | `SSDConfig`, `KVCacheFileHeader` |
| `ssd_model.h/cpp` | `SSDModel` class (~820 lines) |
| `llama3_2_ssd.h` | Model spec for Llama 3.2 3B SSD |
| `llama3_2_ssd_example.cpp` | Interactive chat executable |

### Generation phases

1. **Prefill (AR-128)** — Standard chunked prefill with custom mask skipping
   forecast prefix KV. Position IDs offset by `-forecast_prefix`.

2. **Initial decode (AR-32)** — First token + 2 forecast tokens. Commits only the
   real token's KV. Forecast logits seed the first draft tree.

3. **SSD loop (AR-32)** — Each iteration: build draft tree (CPU top-k) →
   concatenate with forecast tokens → one forward pass → verify → selective KV
   update → build next tree from accepted node's forecast logits.
