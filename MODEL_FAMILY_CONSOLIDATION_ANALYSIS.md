# Per-Family Runtime Consolidation — Implementation Record

**Status:** Implemented (LLM families + Qwen2.5-VL family).
**Last update:** 2026-05-20

This document records the consolidation that replaced per-variant `models/<name>/<name>.h`
files with one runtime per *model family* driven by the standard QAIRT distributed
modelfile bundle (`config.json` + `metadata.json` + `tokenizer.json`).

---

## 1. End state

```
Family runtime                  Variants it serves
────────────────                ──────────────────
models/llama3/llama3.h          Llama 3 (8B-instruct, elyza-JP, taide), 3.1 (8B, sea-lion), 3.2 (1B, 3B)
models/qwen3/qwen3.h            qwen3_4b, qwen3_4b_instruct_2507, qwen3_8b, qwen3_4b_xtensor
models/qwen2_5/qwen2_5.h        qwen2_5_7b_instruct
models/falcon3/falcon3.h        falcon3_7b_instruct
models/phi3_5/phi3_5.h          phi_3_5_mini_instruct
models/qwen2_5_vl/qwen2_5_vl.h  qwen2_5_vl_7b (vision encoder + LLM)
models/llama3_2_ssd/…           llama3_2_3b_ssd  (UNCHANGED — separate runtime per spec)
```

`llama3_1/` and `llama3_2/` directories were deleted; their example executables moved
into `models/llama3/`. The registry still exposes every variant ID; they all map to
the same family factory.

**Bringing up a new variant** of an already-supported family no longer requires any
C++ changes: drop the standard QAIRT bundle (`config.json`, `metadata.json`,
`tokenizer.json`, `*.bin`, `htp_backend_ext_config.json`) into `modelfiles/<name>/`
and add an entry in `llm_model_registry.h` pointing at the family factory.

---

## 2. New core utility — `core/{include,src}/llm/llm_spec_loader.{h,cpp}`

Single new file pair in `core/`. Has no model-specific code.

### `parseHFConfig(bundle_dir) → ParsedHFConfig`

Reads HuggingFace-style `config.json`. Captures every field the runtime needs:

| Field | HF key |
|---|---|
| `model_type` | `model_type` |
| `hidden_size`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size`, `num_hidden_layers` | direct |
| `max_position_embeddings` | direct |
| `rope_theta` | direct |
| `rope_scaling` (tagged variant: `StandardRope` / `Llama3RopeScaling` / `LongRopeScaling` / `PartialRopeScaling`) | `rope_scaling.{rope_type,type}` plus per-variant fields |
| `bos_token_id`, `pad_token_id`, `eos_token_ids` | direct (eos may be int or list) |
| `mrope_section` (VLM) | `rope_scaling.mrope_section` |
| `vision_start_token_id`, `vision_end_token_id`, `image_token_id`, `video_token_id` (VLM) | direct |

### `parseQAIRTMetadata(bundle_dir) → ParsedQAIRTMetadata`

Reads `metadata.json`. Auto-detects two schemas:

**LLM schema** — keys are `[<phase>_]ar<N>_cl<M>_<i>_of_<T>`:
- AR / CL / shard / total parsed by regex; phase prefix optional but consistent
- Walks one representative `(prefill_AR, smallest_CL)` slice per shard
- For each shard: first non-special input → `in_state_name`, first non-special output → `out_state_name` (slashes/dots → underscores), `past_key_<N>_in` keys → contiguous KV layer range
- `seq_len_prefill` = max AR; `seq_len_decode` = min AR
- `context_lengths` = sorted set of all `cl<N>` values
- `graph_name_pattern` = `"{phase}_ar{ar}_cl{cl}_{shard}_of_{total}"` or without phase prefix

**VLM schema** — keys are bare `partN_of_M.bin` plus optional `vision_encoder.bin`:
- Shard wiring + KV ranges parsed from each `partN_of_M.bin` entry the same way
- AR is fixed: prefill = 128, decode = 1 (per Qwen2.5-VL convention; the bundle's metadata doesn't unambiguously specify them)
- `context_lengths` read from top-level `context_lengths` array, or from `genie.context_lengths` (Qwen2.5-VL nests it)
- `graph_name_pattern` = empty (bundles with bare `partN_of_M` keys have no name pattern)
- `vision_encoder_graph` = `"vision_encoder.bin"` when present
- `vision_preprocessing` block (image_width/height, patch_size, temporal_patch_size, spatial_merge_size, normalize_mean/std) read from top-level or from `genie.vision_preprocessing`

### `buildSpecFromConfig(hf, meta) → LLMSpec`

Pure composition; no surprises.

### `makeRoPEProvider(hf) → unique_ptr<InputProvider>`

Dispatches on the `RopeScaling` variant:
- `StandardRope` → `RoPEInputProvider(head_dim, theta)`
- `LongRopeScaling` → `LongRoPEInputProvider(head_dim, theta, long_factor, max_pos, original_max_pos)` — replaces phi3_5's previously hardcoded `kExtFactors`
- `PartialRopeScaling` → `PartialRoPEInputProvider`
- `Llama3RopeScaling` → falls back to standard provider with an INFO log (no dedicated `Llama3RoPEInputProvider` exists in core today; current Llama 3 bundles ship pre-baked RoPE tables, so this matches today's behavior)

### `makeEmbeddingProvider(hf, meta) → unique_ptr<InputProvider>`

Branches on shard 0's input name:
- `input_ids` → `TokenIdInputProvider("input_ids", pad)` where `pad = hf.pad_token_id` ?? `eos[0]` ?? `0`
- `input_embeds` / `inputs_embeds` → `EmbeddingInputProvider(name)` (CPU-side embedding lookup)

### `bundleDirOf(model_cfg)` and `modelConfigFromDirectory(dir)`

Helpers. The latter reads `genie_config.json`'s `dialog.engine.model.binary.ctx-bins`
to recover shard ordering, falling back to a sorted glob of `*.bin`. Tokenizer and
HTP-extension paths are hardcoded filenames (`tokenizer.json` / `htp_backend_ext_config.json`).

---

## 3. Family files

Every family file follows the same pattern:

```cpp
namespace geniex::<family> {

inline LLMModel makeModel(const ModelConfig& model_cfg) {
    const auto bundle = bundleDirOf(model_cfg);
    auto       hf     = parseHFConfig(bundle);
    auto       meta   = parseQAIRTMetadata(bundle);

    LLMModel m(buildSpecFromConfig(hf, meta));
    m.addInputProvider(makeEmbeddingProvider(hf, meta));
    m.addInputProvider(makeRoPEProvider(hf));
    return m;
}

inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg,
                                               const ModelConfig& model_cfg) {
    try {
        LLMPipeline pipe;
        if (!pipe.create(<chatTemplate>, makeModel(model_cfg), runtime_cfg, model_cfg))
            return std::nullopt;
        return pipe;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("<family>::makePipeline failed: {}", e.what());
        return std::nullopt;
    }
}

}  // namespace geniex::<family>
```

The only family-level choices encoded in C++ are:

| Family | Chat template | Notes |
|---|---|---|
| `qwen3` | `chatMLTemplate` | |
| `qwen2_5` | `chatMLTemplate` | |
| `llama3` | `llama3ChatTemplate` | covers Llama 3.0/3.1/3.2 |
| `falcon3` | `falcon3ChatTemplate` | template defined in same header |
| `phi3_5` | `phiChatTemplate` | |
| `qwen2_5_vl` | n/a (VLMPipeline owns templating) | see §5 |

Everything else (shapes, RoPE flavor, EOS, shard wiring, KV ranges, CL set, AR set,
mrope sections, vision-token IDs, vision-preprocessing dimensions) is read from the
bundle.

---

## 4. Registry

`models/llm_model_registry.h` keeps every variant ID for discoverability; they all
map to the same family factory:

```cpp
{"qwen3_4b",                   {qwen3::makePipeline}},
{"qwen3_4b_instruct_2507",     {qwen3::makePipeline}},
{"qwen3_8b",                   {qwen3::makePipeline}},
{"qwen2_5_7b_instruct",        {qwen2_5::makePipeline}},
{"phi_3_5_mini_instruct",      {phi3_5::makePipeline}},
{"llama_v3_8b_instruct",       {llama3::makePipeline}},
{"llama_v3_elyza_jp_8b",       {llama3::makePipeline}},
{"llama_v3_taide_8b_chat",     {llama3::makePipeline}},
{"llama_v3_1_8b_instruct",     {llama3::makePipeline}},
{"llama_v3_1_sea_lion_3_5_8b_r", {llama3::makePipeline}},
{"llama_v3_2_1b_instruct",     {llama3::makePipeline}},
{"llama_v3_2_3b_instruct",     {llama3::makePipeline}},
{"llama_v3_2_3b_instruct_ssd", {llama3_2_3b_ssd::makePipeline}},  // unchanged
{"falcon_v3_7b_instruct",      {falcon3::makePipeline}},
```

`vlm_model_registry.h`:

```cpp
{"qwen2_5_vl_7b_instruct", {qwen2_5_vl::makePipeline}},
```

---

## 5. VLM family — `models/qwen2_5_vl/`

The VLM consolidation matches the LLM one but has additional moving parts:

**Removed from family file** (now read from bundle):

| Was hardcoded | Source |
|---|---|
| `kHiddenSize=3584`, `kNumHeads=28`, `kNumKVHeads=4`, `kHeadDim=128`, `kNumLayers=28`, `kVocabSize=152064`, `kRopeTheta=1e6` | `config.json` |
| `kImageHeight=336`, `kImageWidth=504`, `kPatchSize=14`, `kTemporalPatchSize=2`, `kSpatialMergeSize=2` | `metadata.json` `vision_preprocessing` |
| `kVisionStartTokenId=151652`, `kVisionEndTokenId=151653` | `config.json` `vision_start_token_id` / `vision_end_token_id` |
| `mRoPESection() = {16,24,24}` | `config.json` `rope_scaling.mrope_section` |
| Shard wiring (`inputs_embeds → add_13335 → … → logits`) and KV ranges `[0..5][6..11][12..17][18..23][24..27]` | `metadata.json` `model_files["partN_of_5.bin"]` |
| `context_lengths = {512, 1024, 2048}` | `metadata.json` `genie.context_lengths` |

**Kept as family constants** (not present in any bundle file today):
- `kVitWindowSize = 112` — windowed-attention size in pixels
- `kVitRopeDim = 40` — vision-tower rotary dim
- `kVitRopeTheta = 10000.0f` — vision-tower RoPE base
- `kDefaultImageTokenId = 151655` — fallback when `config.json` has `image_token_id: null` (current Qwen2.5-VL configs do)

These describe the qwen2-vl ViT itself; they're invariant across any Qwen2.5-VL bundle and would only change if the family architecture changed.

`Qwen25VLVisionEncoder` is now configured at runtime via
`setPreprocessing(meta.vision_preprocessing)` and `setHiddenSize(hf.hidden_size)`
before `initialize()`. `Qwen25VLModel` takes its `LLMSpec` by-value through the ctor
and is wired with `setVisionTokenIds(hf.vision_start_token_id, image_token_id)` and
`setSpatialMergeSize(meta.vision_preprocessing->spatial_merge_size)` before init.

The namespace was renamed from `qwen2_5_vl_7b` (a single variant) to `qwen2_5_vl`
(a family). Callers updated: `models/qwen2_5_vl/qwen2_5_vl_7b_example.cpp`,
`qwen2_5_vl_7b_pipeline.cpp`, `tests/vlm.cpp`, `models/vlm_model_registry.h`.

---

## 6. Examples

Every per-variant `<variant>_example.cpp` was updated with a one-line change:
`geniex::<old_variant_ns>::makeModel()` → `geniex::<family>::makeModel(model_cfg)`.

Build targets and registry IDs are unchanged, so `cmake --target qwen3_4b` etc.
still work and downstream consumers don't need to update anything.

`models/llama3_1/` and `models/llama3_2/` were deleted; their seven example .cpp files
moved into `models/llama3/`, with `models/llama3/CMakeLists.txt` updated to enumerate
all seven targets in a single `foreach`.

The continuous-batching helpers (`examples/continuous_batching/qwen3/qwen3_cb.h`,
`examples/continuous_batching/llama3_2/llama3_2_cb.h`) were updated to take a
`ModelConfig` and build their CB-flavored spec via `parseHFConfig` + `parseQAIRTMetadata`.

---

## 7. Phi3.5 LongRoPE

`phi3_5.h` previously contained 48 hardcoded `kExtFactors` values. Those are now
read from `config.json`'s `rope_scaling.long_factor` array via the
`LongRopeScaling` variant in `ParsedHFConfig`, and `makeRoPEProvider` constructs
`LongRoPEInputProvider` with them. The hardcoded constants were deleted.

---

## 8. Verification

A `tests/spec_loader_dump` executable parses any bundle directory and prints the
resulting `ParsedHFConfig` / `ParsedQAIRTMetadata` / composed `LLMSpec`. All three
shipped bundles parse cleanly and the resulting specs match the previously hardcoded
ones byte-for-byte:

```
[OK] modelfiles/qwen3_4b
[OK] modelfiles/qwen3_4b_instruct_2507
[OK] modelfiles/qwen2_5_vl_7b
```

Full release build (`cmake --build build --config Release -j8`) succeeds for every
target: `geniex_core`, `geniex_vlm`, all 7 llama executables, qwen3/qwen2_5/falcon3/
phi3_5 examples, qwen2_5_vl example + pipeline, qwen3_cb, llama3_2_cb, llama3_2_3b_ssd,
llm_test, vlm_test, spec_loader_dump.

---

## 9. What we deliberately did *not* read

`genie_config.json` carries fields that are not strictly necessary for spec
construction and were left out of the loader for now:

- `dialog.type` — picks decoding strategy (basic / ssd-q1 / spd / kv-share / multistream / eaglet / lade). Today the registry encodes the family→strategy choice; reading this field would let a bundle self-declare it.
- `dialog.embedding.{lut-path, size, datatype}` — automatically populates `ModelConfig::embedding_path` for VLM/8B-LLM bundles.
- `dialog.sampler.{temp, top-k, top-p, seed}` — default `GenerationConfig`.

These are candidate follow-ups; the rest of `genie_config.json` duplicates fields
already in `config.json` (rope_theta, head_dim, eos) or carries QNN runtime tuning
parameters that don't belong in `LLMSpec`.

`tokenizer_config.json` is also not consumed; the standalone `Tokenizer` already
loads it transitively via `tokenizer.json`.

---

## 10. Engineering principles

The constraint that `core/` must remain model-agnostic
([engineering-principles.md](.claude/rules/engineering-principles.md) rule 1) is
satisfied: `llm_spec_loader.{h,cpp}` is a generic HuggingFace-format / QAIRT-format
reader, used by every family in `models/`. It depends only on
`qnn-api/src/utils/detail/json.hpp` (already vendored for HTP backend extension config).

`llama3_2_ssd` keeps its own runtime, as the SSD variant has structurally different
decode behavior (AR-32 verification with forecast prefix and tree decoding). Per the
current registry, it remains a separate `makePipeline` factory.
