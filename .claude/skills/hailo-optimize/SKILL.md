---
name: hailo-optimize
description: Optimize and quantize a parsed Hailo HAR using the Hailo Dataflow Compiler (DFC). Auto-loads a matching ALLS model script from the Model Zoo, prepares calibration data (with the correct normalization alignment), runs optimization on GPU when available, and gates on the post-optimize SNR.
argument-hint: <parsed.har> [hw_arch]
allowed-tools: Read, Write, Bash, Glob, Grep
---

# Hailo Optimize

Second stage of the DFC flow: take a parsed `.har` from `/hailo-parse` and produce a quantized `<model_name>_optimized.har`. This step applies a **model script** (ALLS commands — normalization, optimization flavor, NMS postprocess, fine-tune, etc.) and runs the DFC's quantization with a calibration set.

This skill is shipped with the Hailo Model Zoo. It assumes the DFC wheel (`hailo_sdk_client`) is installed in the active virtualenv.

## Inputs

| Arg | Required | Notes |
|---|---|---|
| `<parsed.har>` | yes | Output of `/hailo-parse`. |
| `[hw_arch]` | no | One of `hailo10h` (default), `hailo15h`, `hailo15l`. |

Follow-up questions: calibration source (numpy `.npy` / `.npz`, image directory, or list of image paths), and whether to use a custom ALLS instead of the auto-matched one.

## Workflow

### 1. Resolve inputs

- Verify `<parsed.har>` exists.
- `model_name` = HAR filename stem (strip a trailing `_parsed` if present).
- `hw_arch` defaults to `hailo10h`.
- Confirm `hailo_sdk_client` import works.

### 2. Auto-match ALLS (silent)

Look up an ALLS file in this priority order:

1. `hailo_model_zoo/cfg/alls/<hw_arch>/performance/<model_name>.alls`
2. `hailo_model_zoo/cfg/alls/<hw_arch>/base/<model_name>.alls`
3. `hailo_model_zoo/cfg/alls/generic/<model_name>.alls`

`hailo10h` has no dedicated subdir — for it, only `generic/` is consulted.

Apply the same name-matching rules as `/hailo-parse` (case-insensitive, `-`↔`_`, version-suffix stripping).

On a hit: load the file as the model script and log `Using Model Zoo ALLS: hailo_model_zoo/cfg/alls/<...>/<name>.alls`.

On a miss: compose a minimal default and **show it to the user before applying**:

```
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
model_optimization_flavor(optimization_level=2)
```

(Use the `mean_list` / `std_list` from the matched parser YAML, if `/hailo-parse` passed them along; otherwise the 0/255 defaults assume image input in 0–255 range.)

### 3. Model-script command reference

The user may want to extend the auto-matched ALLS. Most-common commands (lifted from real Model Zoo ALLS files):

| Command | Purpose |
|---|---|
| `normalization1 = normalization([means], [stds])` | Fold normalization into the network. |
| `model_optimization_flavor(optimization_level=0..4, compression_level=0..5)` | Compilation-speed vs accuracy and 4-bit weight aggressiveness. `compression_level=5` ≈ 100% 4-bit weights (most aggressive); `0` disables auto-4-bit. |
| `model_optimization_config(calibration, batch_size=N, calibset_size=M)` | Calibration knobs. |
| `change_output_activation(<layer>, sigmoid)` | Common for YOLO output convs. |
| `nms_postprocess("<path-to-nms.json>", meta_arch=yolov5\|yolov8\|..., engine=cpu)` | Wire NMS in as a CPU postprocess. NMS configs ship in `hailo_model_zoo/cfg/postprocess_config/`. |
| `pre_quantization_optimization(activation_clipping, layers=[...], mode=percentile, clipping_values=[0.01, 99.99])` | Tame outliers before quantization. |
| `post_quantization_optimization(finetune, policy=enabled, learning_rate=2.5e-5)` | Quantization-aware fine-tune. |
| `post_quantization_optimization(bias_correction, policy=enabled)` | Per-channel bias correction. |
| `quantization_param(<layer>, precision_mode=a16_w16)` | Promote a noisy layer to 16-bit. |
| `input_conversion(yuv_to_rgb)` / `resize(resize_shapes=[H,W])` | Preprocessing fused into the network. |

### 4. Calibration data prep

Accept any of:

- A `.npy` or `.npz` array of shape `(N, H, W, C)`.
- A directory of images (`*.jpg`, `*.jpeg`, `*.png`, `*.bmp`).
- A text file listing image paths (one per line).

Auto-convert image inputs into the `(N, H, W, C)` numpy array. Resize to the model's input HxW (read from the HAR or ask the user), keep channels in the order the network expects, and use ≥ 1024 samples when available. Always check the statistics of the calibration set (min/max/mean/etc) and print it to the user.

#### Critical — normalization alignment

Inspect the chosen ALLS for a `normalization(...)` line:

- **If `normalization` is present** → the network normalizes on-device. The calibration set MUST be **raw, un-normalized** (e.g., `uint8` in 0–255 for images). Do **not** subtract mean / divide std on the host.
- **If no `normalization` line** → the user is expected to pre-normalize the calibration data; apply the user-supplied mean/std (or warn if none provided).

Mismatching this is one of the most common customer pitfalls and silently destroys accuracy. The skill verifies it explicitly and refuses to proceed if a normalization layer is in the ALLS but the supplied calibration data is already normalized (heuristic: float dtype with values in `[-3, 3]` or similar).

### 5. GPU detection and TF/GPU pitfalls

Optimization is GPU-heavy (fine-tune, bias correction, percentile clipping). Before running:

```bash
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null
```

- If a GPU is visible and free, set `CUDA_VISIBLE_DEVICES=<idx>` and proceed.
- If `nvidia-smi` is missing or no GPU is free, **warn the user** that optimization will run on CPU and may take much longer (hours instead of minutes for large models). Offer to lower `optimization_level` or `calibset_size` to compensate, and confirm before continuing.

**Common GPU-path pitfalls** (DFC docs `model_optimization.rst`):

- The `hailo_sdk_client` must be imported **before** TensorFlow for GPU support to work correctly. The skill's script always imports it first.
- If optimization fails with VRAM allocation errors, set `HAILO_SET_MEMORY_GROWTH=false` in the environment and re-run.

### 6. Sanity check with SDK_FP_OPTIMIZED before quantizing

Before calling `runner.optimize(...)` (which can take hours), run a quick `InferenceContext.SDK_FP_OPTIMIZED` emulation pass on a handful of calibration samples. This applies the model script *without* quantizing and is the fastest way to catch normalization / preprocessing / NMS-config mismatches:

```python
from hailo_sdk_client import InferenceContext

with runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
    out = runner.infer(ctx, calib_dataset[:4])
# compare `out` to the original framework's output on the same 4 samples
```

If the outputs diverge wildly here, fix the model script before paying for a full optimization run.

### 7. Run optimization (full quantization)

```python
from hailo_sdk_client import ClientRunner
import numpy as np

runner = ClientRunner(har="<parsed.har>")
runner.load_model_script(open("<model_name>.alls").read())

calib_dataset = np.load("<calib.npy>")  # or constructed in step 4
runner.optimize(calib_dataset)

runner.save_har("<model_name>_optimized.har")
```

Capture stdout — the SNR summary printed at the end is parsed in step 8.

### 8. Validate quality from SNR console output

The DFC prints per-output SNR (in dB) at the end of `runner.optimize(...)`. Parse those lines and gate on a threshold:

- **Default threshold: ≥ 10 dB** per output. This matches the DFC convention — the CLI tools flag layers with SNR < 10 dB as "most sensitive" (`command_line_tools.rst`). Tighten the threshold (e.g. ≥ 16 dB) for accuracy-critical deployments; loosen it for very lossy tasks.
- All outputs above threshold → report success and continue.
- One or more outputs below threshold → suggest, in order:
  1. Raise `model_optimization_flavor(optimization_level=2 → 3)`.
  2. Add `post_quantization_optimization(finetune, policy=enabled, learning_rate=2.5e-5)` if not already enabled.
  3. Add `post_quantization_optimization(bias_correction, policy=enabled)`.
  4. Run `runner.analyze_noise(...)` (DFC tutorial `DFC_5_Layer_Noise_Analysis_Tutorial.ipynb`, extract via `hailo tutorial`) to identify the worst layers.
  5. Promote the worst layers via `quantization_param(<layer>, precision_mode=a16_w16)`.
  6. **Last resort — Quantization-Aware Training (QAT)**: see `DFC_6_QAT_Tutorial.ipynb`. QAT typically recovers the most accuracy when the simpler escalations above fail, but it requires the original training pipeline (data + loss + optimizer) — only suggest this when the user has retraining infrastructure available.

### 9. Hand-off

> Optimization complete. Next: run `/hailo-compile <model_name>_optimized.har [hw_arch]` to compile to a HEF.

## Notes

- The skill never modifies the DFC wheel or Model Zoo configs — it only reads YAMLs and writes a `.har` in the current directory.
- All Model Zoo paths are relative to the Model Zoo repo root.
- The skill never edits ALLS files in `hailo_model_zoo/cfg/alls/` — if the user wants changes, they're applied to a working copy of the script in the current directory.
- Tutorials: `DFC_2_Model_Optimization_Tutorial.ipynb` (full reference) and `DFC_5_Layer_Noise_Analysis_Tutorial.ipynb` (debugging quantization). Extract with `hailo tutorial`.
- DFC User Guide: https://hailo.ai/developer-zone/.
