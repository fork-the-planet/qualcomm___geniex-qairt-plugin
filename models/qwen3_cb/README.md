# Continuous Batching for Qwen3-4B on QAIRT

Continuous batching enables serving multiple concurrent user sessions through a single `batch_size=1` QNN model binary by concatenating inputs along the sequence dimension instead of stacking along a batch dimension.

## Why Concatenate Instead of Stack?

On GPUs, batching stacks inputs along a batch dimension because the hardware is compute-bound and exploits batch parallelism. QNN inference on Snapdragon NPUs is **memory-bandwidth-bound**, so there is no performance difference between stacking and concatenating. Concatenation has a major advantage: it works with existing `batch_size=1` model exports without re-compilation.

A standard single-session decode step wastes 127 of the 128 input slots on padding. With continuous batching, those slots serve other sessions.

## Architecture

The implementation follows a three-component design:

### 1. Scheduler

Manages the lifecycle of concurrent sessions. Each session progresses through `WAITING -> RUNNING -> COMPLETED` states.

| Method | Purpose |
|---|---|
| `addSession` | Register a new session with its tokenized query |
| `getNextBatch` | Select which sessions to include in the next forward pass, packing tokens up to `seq_len` |
| `updateSession` | Advance a session's `processed_length` after a forward pass |
| `completeSession` | Mark a session as done (EOS or max tokens reached) |
| `removeSession` | Evict a completed session |

### 2. KV Cache Manager

Tracks per-session KV cache segments within the shared contiguous buffer that QNN allocates. The KV cache layout concatenates all sessions along the sequence dimension:

```
|<-- Session A (len=20) -->|<-- Session B (len=8) -->|   free   |
```

| Method | Purpose |
|---|---|
| `allocate` | Reserve a segment for a new session |
| `extend` | Grow a session's segment after processing new tokens |
| `release` | Free a completed session's segment |
| `compact` | Defragment by shifting segments left to fill gaps |
| `shiftForGrowth` | Shift segments right before a step so each session has room to grow |
| `getPositionIds` | Build per-session position IDs (each session counts from its own KV length) |
| `getAttentionMask` | Build block-diagonal causal mask so sessions never attend to each other |

The attention mask is the critical correctness component. Without it, session A's tokens would attend to session B's KV cache, producing garbage. The mask ensures each session sees only its own history:

```
         KV cache columns     Current input columns
         A-seg    B-seg       A-input    B-input
A row: [ allow    block       causal     block   ]
B row: [ block    allow       block      causal  ]
```

### 3. Next Token Extraction

After the forward pass, the output logits contain interleaved predictions for all sessions. `extractNextTokens` splits them by session boundaries and takes the last-position logit for each session as its next-token prediction.

## CBLLMModel

`CBLLMModel` subclasses `LLMModel` and overrides `generate()`. The main loop in `generateBatch()` composes the three components:

1. **Scheduler** selects sessions for this step
2. **KV Cache Manager** allocates/shifts segments, builds attention mask and position IDs
3. All shards execute on the concatenated input (always using the prefill graph, phase=0)
4. KV outputs are copied into each session's segment
5. **Scheduler** updates processed lengths
6. **Next Token Extraction** samples one token per session that finished prefill
7. **KV Cache Manager** releases completed sessions and compacts

The model reuses the same `qwen3_4b_instruct_2507_aihub` architecture spec and model weights. No model re-export is needed.

## Files

| File | Purpose |
|---|---|
| `qwen3_cb.h` | Header-only: `Scheduler`, `KVCacheManager`, `extractNextTokens`, `CBLLMModel`, and model spec |
| `qwen3_cb_example.cpp` | Interactive example: enter multiple prompts, type `go` to run them concurrently |
| `CMakeLists.txt` | Build target `qwen3_cb` |

## Usage

```bash
# Build
cmake -B build -A ARM64
cmake --build build --config Release --target qwen3_cb -j32

# Run (from project root)
./build/bin/Release/qwen3_cb.exe --max-tokens 50 --verbose
```

At the interactive prompt, enter one query per line and type `go` to start generation:

```
Prompt 1 (or 'go'/'exit'): Hello
Prompt 2 (or 'go'/'exit'): What is 2+2?
Prompt 3 (or 'go'/'exit'): go
```

## Performance

Measured on Snapdragon X Elite (Qwen3-4B, 128-token prefill window):

| Sessions | Total tokens | Throughput |
|---|---|---|
| 1 | 11 | ~8.2 tok/s |
| 2 | 19 | ~13.2 tok/s |
| 3 | 55 | ~12.7 tok/s |

Multi-session throughput improves ~60% over single-session because decode steps fill the 128-token window with real tokens from multiple sessions instead of padding.
