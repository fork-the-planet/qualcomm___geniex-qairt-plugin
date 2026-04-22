# Qwen2.5-VL-7B-Instruct

Vision-language inference for the Qwen2.5-VL-7B-Instruct model on Qualcomm NPU.

## Key notes

- LLM context length is *2048*, each image consumes ~218 / 2048 ≈ 11 % of the context. Two images leave ~1600 tokens for text and generation; beyond four images the prompt alone is close to filling the KV cache.

- The vision_encoder graph is compiled for a **fixed** 24×36 patch grid (336×504 px). Every input image is force-resized to 336×504 — aspect ratio is not preserved.

## End-to-end workflow

```
image paths + text
  │
  ▼  Qwen2VLProcessor (fixed 336×504)
      resize → CLIP normalize → HWC→CHW → temporal tile →
      pixel_values [N_img·864, 1176] + input_ids with 216·N_img <|image_pad|>
  │
  ▼  Qwen25VLVisionEncoder::encode
      per image:  pixel_values + rotary cos/sin + window/full masks
                  → execute vision_encoder.bin → 216 tokens × 3584
      concat → [N_img·216, 3584]
  │
  ▼  VLMModel::generate
      text embeds ← PrecomputedEmbeddingProvider
      maskedScatter: vision_embeds → rows where input_ids == <|image_pad|>
      MRoPEInputProvider: 3D positions from image_grid_thw
  │
  ▼  LLMModel: 5-shard prefill (AR=128, chunked) + decode (AR=1)
      → streamed output tokens
```
