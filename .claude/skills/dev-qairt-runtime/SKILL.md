---
name: dev-qairt-runtime
description: Reference for developing against the QAIRT/QNN runtime — graph execution, KV cache, tensor I/O, multi-shard wiring
user-invocable: true
---

# QAIRT Runtime Development Reference

Use this skill when writing or modifying runtime code that interacts with QNN graphs, KV cache, or the inference loop.

## Graph execution model

Each compiled context binary (`.bin`) contains one or more **graphs**. A graph is a single forward pass for a fixed `(sequence_length, context_length)` configuration.

```
context_binary.bin
  ├── prompt_ar128_cl512_1_of_2    (prefill, 128 tokens, CL=512, shard 1)
  ├── prompt_ar128_cl512_2_of_2    (prefill, 128 tokens, CL=512, shard 2)
  ├── token_ar1_cl512_1_of_2       (decode, 1 token, CL=512, shard 1)
  └── token_ar1_cl512_2_of_2       (decode, 1 token, CL=512, shard 2)
```

### Graph naming pattern

`LLMModel::onInitialized` parses graph names with regex
`(?:[A-Za-z]+_)?ar(\d+)_cl(\d+)_(\d+)_of_(\d+)`, accepting both prefixed
(`prompt_`/`token_`) and unprefixed forms. Phase is inferred from `ar`
(largest AR ⇒ prefill, smallest ⇒ decode). The CL set, AR set, and
`seq_len_prefill / seq_len_decode` on `LLMSpec` are all populated from
the loaded QNN graph names — nothing to set on the spec.

- `<phase>` — `prompt` (prefill) or `token` (decode), optional
- `ar` — autoregressive length (e.g. 128 or 1)
- `cl` — context length
- `shard` / `total` — shard index and count

## Tensor I/O

### Writing inputs

```cpp
// Type-safe write (handles float32, float16, ufixed8/16, int32 conversion)
graph.write("input_tensor_name", float_ptr);

// Direct buffer access (zero-copy, caller handles dtype)
float* ptr = static_cast<float*>(graph.inputPtr("tensor_name"));
```

### Reading outputs

```cpp
graph.read("output_tensor_name", float_output_ptr);
float* ptr = static_cast<float*>(graph.outputPtr("tensor_name"));
```

### Inspecting tensor metadata

```cpp
auto input_specs = graph.inputSpecs();   // vector<TensorSpec>
auto output_specs = graph.outputSpecs(); // vector<TensorSpec>
// TensorSpec has: name, dtype, shape, quantization params
```

## Multi-shard execution

Large models are split across shards. Each shard processes a subset of layers. Hidden states are passed between shards via **connections**.

```
Shard 1: input_ids → hidden_state_mid
Shard 2: hidden_state_mid → logits
```

The `LLMSpec.shards` field defines this wiring:
```cpp
.shards = {
    ShardSpec{"input_ids", "hidden_state_mid"},
    ShardSpec{"hidden_state_mid", "logits"},
}
```

The runtime automatically routes shard N's output tensor to shard N+1's input tensor.

## KV cache management

### Memory layout

```
[num_layers, num_kv_heads, context_length, head_dim]
```

Pre-allocated at max CL. Unused positions are zero-padded and masked via the attention mask.

### KV state block definition

```cpp
.state_blocks = {
    makeKVOnlyStateBlock({
        std::nullopt,           // shard 0: no KV (embedding-only)
        LayerRange{0, 15},      // shard 1: layers 0-15
        LayerRange{16, 31},     // shard 2: layers 16-31
    }),
}
```

### KV tensor naming

```cpp
.kv_key_in_pattern    = "past_key_{}_in"      // {} = layer index
.kv_value_in_pattern  = "past_value_{}_in"
.kv_key_out_pattern   = "past_key_{}_out"
.kv_value_out_pattern = "past_value_{}_out"
```

### Context-length promotion

When a model has multiple CL variants (e.g. 512, 1024, 2048, 4096):
1. Start with smallest CL that fits the prompt
2. When sequence grows past current CL, promote to next variant
3. KV cache is reshaped (expand operation: process last row to first, zero-pad new columns)

## Attention mask construction

The mask must handle three concerns simultaneously:
1. **Causal masking** — token at position i can only attend to positions <= i
2. **Padding masking** — padded positions in the current input chunk are masked out
3. **KV cache masking** — unfilled positions in the KV buffer are masked out

For prefill with padding:
```
Prompt: [A, B, C, PAD, PAD, ...] (padded to 128)
Mask row for token C (position 2): [1, 1, 1, 0, 0, ..., 0]
                                    ↑attend  ↑pad  ↑unfilled KV
```

## Prefill/decode loop

```
1. Tokenize prompt → token_ids
2. For each 128-token chunk:
   a. Prepare inputs (embeddings/token_ids, RoPE, attention mask)
   b. Execute prefill graph for each shard sequentially
   c. Update KV cache position counter
3. Decode loop (until EOS or max_length):
   a. Prepare inputs for single token
   b. Execute decode graph for each shard
   c. Sample next token from logits
   d. Increment position
   e. Promote CL if needed
```

## Weight sharing

Multiple graph variants (different AR lengths, different CLs) share the same weight tensors in memory. Only activation buffers and KV cache differ. This is managed automatically by QNN when graphs are loaded from the same context binary.

## Shared I/O buffers

Prefill (ARN) and decode (AR1) graphs share input/output buffers since they never execute concurrently. Writing to one graph's input buffer modifies the other's. This is intentional and saves memory.

## Common dtype handling

| Export type | Tensor dtype | Notes |
|-------------|--------------|-------|
| Genie w4 | float16 | Weights quantized, activations in fp16 |
| Genie w8 | float32 | Standard precision |
| AI Hub | varies | Check `TensorSpec.dtype` at runtime |

`Graph::write(float*)` and `Graph::read(float*)` handle conversion transparently. You always work with float32 on the CPU side.

## Performance tips

- Minimize CPU-NPU transfers: keep `InputProvider` logic efficient
- Match CL to workload: use Multi-CL exports to avoid over-allocating KV buffers
- Profile with `--verbose` flag on example executables for timing breakdown
- TTFT dominated by prefill speed; decode throughput typically 20-50 tok/s on Snapdragon 8 Gen 3
