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
| `model_optimization_flavor(optimization_level=0..4, compression_level=0..5)` | Optimization method and 4-bit weight aggressiveness — **not a monotone accuracy ladder**, see below. `compression_level=5` ≈ 100% 4-bit weights (most aggressive); `0` disables auto-4-bit. |
| `model_optimization_config(calibration, batch_size=N, calibset_size=M)` | Calibration knobs. |
| `change_output_activation(<layer>, sigmoid)` | Common for YOLO output convs. |
| `nms_postprocess("<path-to-nms.json>", meta_arch=yolov5\|yolov8\|..., engine=cpu)` | Wire NMS in as a CPU postprocess. NMS configs ship in `hailo_model_zoo/cfg/postprocess_config/`. |
| `pre_quantization_optimization(activation_clipping, layers=[...], mode=percentile, clipping_values=[0.01, 99.99])` | Tame outliers before quantization. |
| `post_quantization_optimization(finetune, policy=enabled, learning_rate=2.5e-5)` | Quantization-aware fine-tune. |
| `post_quantization_optimization(bias_correction, policy=enabled)` | Per-channel bias correction. |
| `quantization_param(<layer>, precision_mode=a16_w16)` | Promote a noisy layer to 16-bit. |
| `input_conversion(yuv_to_rgb)` / `resize(resize_shapes=[H,W])` | Preprocessing fused into the network. |

#### `optimization_level` is a choice of method, not a monotone ladder

Each level runs a **different** optimization algorithm — a higher number is not simply "more" of what a lower number does:

| Level | Algorithm |
|---|---|
| 0 | Equalization only |
| 1 | Equalization + iterative bias correction |
| 2 | Equalization + QAT fine-tune (4 epochs, 1024 calibration images) |
| 3 | Equalization + AdaRound (320 epochs, 256 images) |
| 4 | Equalization + AdaRound (320 epochs, 1024 images) |

Levels 3 and 4 **replace** the fine-tune with AdaRound — they do not add AdaRound on top of it. Moving a script from level 2 to level 3 while also disabling AdaRound therefore gets *weaker* optimization, not stronger; never do both at once, and don't treat 2 → 3 as a generic "try harder" escalation.

**Real defaults** matter whenever a script omits `model_optimization_flavor(...)` entirely: `optimization_level` defaults to 2 only when a GPU is available **and** the calibration set has at least 1024 images; it drops to 1 with a smaller calibration set, and to 0 with no GPU at all. `compression_level` defaults to 1 (roughly 20% of weights auto-converted to 4-bit) when the resolved optimization level is above 1 and the network has more than 20M parameters, otherwise it defaults to 0. Because these defaults depend on the machine and the calibration set actually used, pin both `optimization_level` and `compression_level` explicitly in any model script whose accuracy numbers must reproduce on another machine.

`post_quantization_optimization(adaround, policy=disabled)` is a no-op at level 2 or below — AdaRound is only scheduled starting at level 3, and its policy already defaults to disabled. Don't add this line, and don't carry it over from a sibling model's script.

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

`runner.optimize(...)` (and `hailomz optimize`) writes its output HAR under a name derived from the network, in the current directory — the same name the parse stage wrote to. Copy the parsed HAR aside first (e.g. `<net>_parsed.har`) or it gets overwritten:

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
- One or more outputs below threshold → escalate in order, and stop as soon as SNR recovers:
  1. Promote the starved layer to 16-bit: `quantization_param([<layer>], precision_mode=a16_w16)`.
  2. Add fine-tune epochs: `post_quantization_optimization(finetune, policy=enabled, epochs=8)`. `policy=enabled` is mandatory here — the model-script parser rejects the command without it, even though the flavor-level fine-tune (level 2) enables it implicitly.
  3. Only then consider AdaRound (`optimization_level=3` or `4`) or full QAT with the original training pipeline (data + loss + optimizer) — see `DFC_6_QAT_Tutorial.ipynb`. This is the most expensive step and, per the level table above, replaces rather than adds to whatever fine-tune was running before.

  Do not treat "raise the optimization level" as the first move, and do not assume a higher level adds fine-tune on top of what a lower level already had — see the level table above.

  Run `runner.analyze_noise(...)` (DFC tutorial `DFC_5_Layer_Noise_Analysis_Tutorial.ipynb`, extract via `hailo tutorial`) beforehand to identify which layer to target with step 1.

**hailo10h caveat**: `precision_mode=a16_w8` (and `a16_w4`) on a conv currently crashes the fine-tune stage on hailo10h. Use `a16_w16` instead. Separately, the hardware rejects a 16-to-8 precision transition feeding an 8-bit matmul — if the starved layers are a matmul's input projections, pin the layer *after* the matmul instead; its transition is 8-to-16 and is legal.

If `runner.analyze_noise(...)` itself crashes with a dimension mismatch on a decomposed matmul, disable the checker so optimization can proceed — `model_optimization_config(checker_cfg, policy=disabled)` — and get per-layer SNR another way (e.g. compare native vs quantized outputs directly). This is a known DFC issue with attention blocks whose projections aren't a multiple of the vector width.

**Noise floor**: an SNR or accuracy check on a small validation set (a few hundred images) carries run-to-run noise of order 0.1 metric points, since the calibration draw and fine-tune are both stochastic. A result that sits within noise of a pass/fail gate should be repeated before it's treated as a real pass or fail.

#### Global average pool overflow

A global average pool over a large spatial map (roughly 8k pixels and up — e.g. 72x128) can overflow the accumulator shift and abort numerization. Split it into two pooling stages with:

```
pre_quantization_optimization(global_avgpool_reduction, layers=[<pool>, ...], division_factors=[a, b])
```

The two factors must tile the input spatial size exactly — e.g. 72x128 splits as a 9x16 pool followed by an 8x8 pool. A layer in `layers` that was dead-code eliminated earlier in the graph is silently ignored, so a factor mismatch elsewhere is the more likely cause if the overflow persists.

### 9. Hand-off

> Optimization complete. Next: run `/hailo-compile <model_name>_optimized.har [hw_arch]` to compile to a HEF.

## Notes

- The skill never modifies the DFC wheel or Model Zoo configs — it only reads YAMLs and writes a `.har` in the current directory.
- All Model Zoo paths are relative to the Model Zoo repo root.
- The skill never edits ALLS files in `hailo_model_zoo/cfg/alls/` — if the user wants changes, they're applied to a working copy of the script in the current directory.
- Tutorials: `DFC_2_Model_Optimization_Tutorial.ipynb` (full reference) and `DFC_5_Layer_Noise_Analysis_Tutorial.ipynb` (debugging quantization). Extract with `hailo tutorial`.
- DFC User Guide: https://hailo.ai/developer-zone/.
