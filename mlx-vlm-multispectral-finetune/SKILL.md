---
name: mlx-vlm-multispectral-finetune
description: Use when fine-tuning a vision-language model on multi-spectral satellite imagery (e.g. Sentinel-2, Landsat) using mlx-vlm on Apple Silicon with LoRA
---

# Fine-Tuning mlx-vlm on Multi-Spectral Data

## Overview

Multi-spectral sensors produce N-band TIFFs that VLMs cannot ingest directly. The approach: render selected bands as RGB false-colour composites, then LoRA fine-tune a VLM (Qwen2.5-VL or Qwen3-VL) using mlx-vlm on Apple Silicon. Three stages: **prepare → train → evaluate**.

---

## Stage 1 — Data Preparation: TIFFs → Composite PNGs + JSONL

### Reading multi-spectral TIFFs

Use `tifffile` (primary) or `rasterio` (fallback). `tifffile` may return `(H, W, C)` — normalise to `(C, H, W)`:

```python
import tifffile, numpy as np

def read_tiff(path):
    arr = tifffile.imread(str(path)).astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] <= arr.shape[0]:   # (H,W,C) → (C,H,W)
        arr = arr.transpose(2, 0, 1)
    return arr
    # fallback: rasterio.open(path).read() already returns (C,H,W)
```

### Band selection (Sentinel-2, 0-based indices in EuroSAT_MS TIFFs)

Three composites cover the most discriminative band ranges. Generate all three per TIFF — this triples dataset size (implicit multi-view augmentation) and preserves spectral information that any single composite loses.

| Composite | R / G / B indices | Wavelengths | Highlights |
|---|---|---|---|
| `natural` | 3 / 2 / 1 | B04/B03/B02 | True colour |
| `falsecolor` | 7 / 3 / 2 | B08/B04/B03 | Vegetation, NIR |
| `swir` | 11 / 7 / 3 | B11/B08/B04 | Urban, water, burn scars |

**SWIR choice:** Use B11 (1610 nm) + B08 (broad NIR), not B12 (2190 nm) + B08A (narrow NIR). B11/B08 gives better urban vs. water separation for land cover tasks.

**Unused bands:** B05–B07 (red edge), B09 (water vapour), B10 (cirrus). Add a red-edge composite `(6, 4, 1)` if vegetation class separation is insufficient.

### Band normalisation

Normalise each band independently (not across the composite) to remove atmospheric haze and sensor saturation. Raw values are uint16 reflectance (0–~10 000):

```python
def normalize_band(arr):
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    return np.clip((arr - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)
```

### Resizing

EuroSAT images are 64×64 px. Resize to 224×224 with Lanczos so the VLM image processor receives a usable input. Note: upscaling does not add information — the model still sees the same coarse textures. The Qwen VL processor internally upscales further to ~448×448, producing 256 image tokens per image.

```python
from PIL import Image
rgb = np.stack([normalize_band(data[r]), normalize_band(data[g]), normalize_band(data[b])], axis=-1)
img = Image.fromarray(rgb, "RGB").resize((224, 224), Image.LANCZOS)
img.save(out_path)
```

### Idempotent composite generation

Check all three output PNGs before reading the TIFF — re-running `--prepare` after a crash skips already-processed images:

```python
def make_composites(tif_path, out_dir, stem):
    paths, pending = {}, {}
    for name in COMPOSITES:
        out_path = out_dir / f"{stem}_{name}.png"
        if out_path.exists():
            paths[name] = str(out_path.resolve())
        else:
            pending[name] = out_path
    if not pending:
        return paths                  # all done, skip TIFF read
    data = read_tiff(tif_path)        # only loaded when at least one PNG is missing
    for name, out_path in pending.items():
        r, g, b = COMPOSITES[name]
        ...
```

Also: `Path.glob()` returns a generator — `len(img_dir.glob("*.png"))` raises `TypeError`. Use `sum(1 for _ in img_dir.glob("*.png"))`.

### JSONL row format

The image path appears in **two** places. Both must be the absolute string path — not an integer index:

```json
{
  "images": ["/abs/path/image.png"],
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "image": "/abs/path/image.png"},
      {"type": "text",  "text": "What land cover does this Sentinel-2 image show?"}
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "forest"}]}
  ]
}
```

- `images[0]` — used by the custom dataset `__getitem__` to open PIL images
- `content[0]["image"]` — used by `VisionDataset` if you switch back to it; must be a path string, not `0` (an integer index silently fails)

Write one file per split: `train.jsonl`, `valid.jsonl`, `test.jsonl`. Shuffle rows before splitting to break class-ordering artefacts from directory structure.

---

## Stage 2 — Training

### Load model and local JSONL datasets

`lora.py` CLI only accepts HuggingFace Hub IDs and hardcodes `val_dataset=None`. Call trainer internals directly:

```python
from mlx_vlm import load
from datasets import load_dataset

model, processor = load(
    "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    processor_config={"trust_remote_code": True},  # avoids PyTorch dep in transformers ≥5.5
)
config = model.config.__dict__

# The key in data_files must match the split= argument exactly
train_hf = load_dataset("json", data_files={"train": "data/train.jsonl"}, split="train")
valid_hf = load_dataset("json", data_files={"valid": "data/valid.jsonl"}, split="valid")
# "validation" ≠ "valid" — will raise KeyError at runtime
```

### Custom dataset class (required for Qwen models)

`VisionDataset` hardcodes `images=None` for all `qwen*` model types, silently producing text-only training. Write your own:

```python
from PIL import Image as PILImage
from mlx_vlm.prompt_utils import apply_chat_template as mlx_apply
from mlx_vlm.utils import prepare_inputs
import mlx.core as mx

class MultiSpectralDataset:
    def __init__(self, hf_dataset, config, processor):
        self.dataset, self.config, self.processor = hf_dataset, config, processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        pil_images = [PILImage.open(p) for p in row["images"]]
        question = next(c["text"] for c in row["messages"][0]["content"] if c.get("type") == "text")
        answer   = row["messages"][1]["content"][0]["text"]

        # Use mlx_vlm.prompt_utils.apply_chat_template — inserts exactly 1 image
        # placeholder; prepare_inputs then expands it to the correct token count
        # from actual pixel dims. processor.apply_chat_template pre-computes a
        # different token count that mismatches the vision encoder output.
        user_prompt = mlx_apply(
            processor=self.processor,
            config=self.config,
            prompt=question,
            num_images=len(pil_images),
            add_generation_prompt=True,
            enable_thinking=False,        # Qwen3: suppress chain-of-thought in targets
        )
        full_prompt = user_prompt + answer + "<|im_end|>"

        image_token_index = (
            self.config.get("image_token_index") or self.config.get("image_token_id")
        )
        inputs = prepare_inputs(
            processor=self.processor,
            images=pil_images,
            prompts=[full_prompt],
            image_token_index=image_token_index,
        )

        # Processor returns [1, seq_len] tensors. iterate_batches computes lengths
        # via len(x["input_ids"]) → 1 (batch dim), giving padded_len=33 and
        # silently truncating every sequence, dropping almost all image tokens.
        ids  = inputs["input_ids"]
        mask = inputs.get("attention_mask", mx.ones_like(ids))
        if isinstance(ids,  mx.array) and ids.ndim  == 2: ids  = ids[0]
        if isinstance(mask, mx.array) and mask.ndim == 2: mask = mask[0]

        return {
            "input_ids":      ids,
            "attention_mask": mask,
            "pixel_values":   inputs.get("pixel_values"),
            **{k: v for k, v in inputs.items()
               if k not in ("input_ids", "attention_mask", "pixel_values")},
        }
```

### Apply LoRA and train

```python
import argparse, mlx.optimizers as optim
from mlx_vlm.lora import setup_model_for_training
from mlx_vlm.trainer.sft_trainer import TrainingArgs, train
from mlx_vlm.trainer.utils import print_trainable_parameters
from pathlib import Path

lora_args = argparse.Namespace(
    lora_rank=8, lora_alpha=16, lora_dropout=0.0,
    full_finetune=False, train_vision=False,
)
model = setup_model_for_training(model, lora_args, adapter_path=None)
print_trainable_parameters(model)

adapter_dir = Path("adapters")
adapter_dir.mkdir(exist_ok=True)

train(
    model=model,
    optimizer=optim.Adam(learning_rate=1e-5),
    train_dataset=MultiSpectralDataset(train_hf, config, processor),
    val_dataset=MultiSpectralDataset(valid_hf, config, processor),
    args=TrainingArgs(
        batch_size=4,
        iters=500,          # gradient steps — do NOT .select(range(iters)) on the dataset
        val_batches=25,
        max_seq_length=1024,
        adapter_file=str(adapter_dir / "adapters.safetensors"),  # name is hardcoded internally
        train_on_completions=True,
        assistant_id=77091,
    ),
)
```

---

## Stage 3 — Evaluation

Load test rows directly from JSONL (no HuggingFace wrapper needed for evaluation):

```python
import json
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model, processor = load(
    "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    adapter_path="adapters",                        # directory, not .safetensors path
    processor_config={"trust_remote_code": True},
)
config = load_config("mlx-community/Qwen2.5-VL-3B-Instruct-4bit")

with open("data/test.jsonl") as f:
    test_rows = [json.loads(line) for line in f]

for row in test_rows:
    img_path = row["images"][0]
    question = next(c["text"] for c in row["messages"][0]["content"] if c.get("type") == "text")
    expected = row["messages"][1]["content"][0]["text"]

    prompt = apply_chat_template(processor, config, question,
                                 num_images=1, enable_thinking=False)
    result = generate(model, processor, prompt, img_path,
                      max_tokens=512, verbose=False)
    output = result.text.strip().lower()   # result is GenerationResult, not str

    if "<think>" in output:                # strip residual thinking tokens
        output = output.split("</think>")[-1].strip()

    hit = expected.lower() in output or output in expected.lower()
```

---

## Key Rules (Break These = Silent Wrong Results)

1. **Normalise bands independently**, not across the RGB composite — otherwise bright NIR bands wash out visible channels.
2. **SWIR composite:** use B11/B08/B04 `(11, 7, 3)`, not B12/B08A `(12, 8, 3)` — better urban/water separation.
3. **Image path in JSONL goes in two places** (`images` list and `content` entry) — both must be absolute path strings, not integer indices.
4. **Split name in `data_files` must match `split=`** — `"valid"` ≠ `"validation"`.
5. **Use `mlx_vlm.prompt_utils.apply_chat_template`**, not `processor.apply_chat_template` — only the former inserts 1 placeholder that `prepare_inputs` expands correctly.
6. **Squeeze `input_ids` to `[seq_len]`** — `[1, seq_len]` from the processor causes batch collation to truncate every sequence to 33 tokens.
7. **`iters` = gradient steps, not row count** — never `.select(range(iters))` on the dataset.
8. **Adapter path is a directory**; weights file inside must be named `adapters.safetensors`.
9. **`enable_thinking=False` everywhere for Qwen3** — thinking mode fills the generation budget before producing a label.
10. **Call trainer internals directly**, not `lora_main()` — the CLI hardcodes `val_dataset=None`.
