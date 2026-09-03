---
name: hailo-compile
description: Compile an optimized Hailo HAR into a HEF binary using the Hailo Dataflow Compiler (DFC). Final stage of the DFC flow — takes the quantized HAR from /hailo-optimize and produces a deployable .hef.
argument-hint: <optimized.har> [hw_arch]
allowed-tools: Read, Write, Bash, Glob, Grep
---

# Hailo Compile

Third stage of the DFC flow: take the quantized `<model_name>_optimized.har` from `/hailo-optimize` and produce a `<model_name>.hef` that runs on the Hailo accelerator.

This skill is shipped with the Hailo Model Zoo. It assumes the DFC wheel (`hailo_sdk_client`) is installed in the active virtualenv.

## Inputs

| Arg | Required | Notes |
|---|---|---|
| `<optimized.har>` | yes | Output of `/hailo-optimize`. |
| `[hw_arch]` | no | Read from the HAR if possible; otherwise default `hailo10h`. |

## Workflow

### 1. Resolve inputs

- Verify `<optimized.har>` exists.
- `model_name` = HAR filename stem (strip a trailing `_optimized` if present).
- `hw_arch` is read from the HAR; if unavailable, ask the user (or default to `hailo10h`).

### 2. Run compile

```python
from hailo_sdk_client import ClientRunner

runner = ClientRunner(har="<optimized.har>")
hef = runner.compile()
with open("<model_name>.hef", "wb") as f:
    f.write(hef)
runner.save_har("<model_name>_compiled_model.har")
```

The compiled HAR carries the metadata the profiler and `hailo har extract` consume. The `_compiled_model.har` suffix matches the DFC tutorial (`DFC_3_Compilation_Tutorial.ipynb`) and the `compilation.rst` documentation — downstream tooling expects this convention.

### 3. Verify

- Confirm `<model_name>.hef` exists and report its size and location.
- Inspect the HEF metadata with `hailortcli parse-hef <model_name>.hef` (requires HailoRT installed).
- **Per-layer profiler (static)** — `hailo profiler <model_name>_compiled_model.har` produces a static per-layer report. **Note**: for multi-context models (most large models), this report does **not** include performance / FPS numbers without runtime data.
- **Profiler with runtime data (full FPS / utilization)** — the documented two-step flow for accurate performance on multi-context models:
  ```bash
  hailortcli run2 -m raw measure-fw-actions --output-path runtime.json set-net <model_name>.hef
  hailo profiler <model_name>_compiled_model.har --runtime-data runtime.json --out-path runtime_profiler.html
  ```
- **Smoke test on the device** — `hailortcli run2 set-net <model_name>.hef` runs inference with random inputs as a quick sanity check (verify the exact form with `hailortcli run --help` — `run2` always requires `-m <mode>`).

### 4. Failure triage

If `runner.compile()` fails or returns very poor profiler results, the fix is almost always in the ALLS model script. Re-run `/hailo-optimize` with one or more of the following commands appended to the script, then re-run this skill.

**Extract what the compiler actually used.** Once any compile finishes, you can recover the *exact* ALLS the compiler resolved (with all defaults filled in) for fast deterministic iteration:

```bash
hailo har extract <model_name>_compiled_model.har --auto-model-script-path auto.alls
```

Iterating against `auto.alls` is much faster and more deterministic than re-running optimize+compile from a partial script.

> **Platform target caveat**: if the ALLS contains `platform_param(targets=[ethernet])`, the compiler disables DDR portals, multi-context, and Sequencers. Several of the tricks below (`context_switch_param`, anything DDR-related) are **no-ops** in that case — fix the platform target first or accept the constraint.

| Symptom | ALLS command to try |
|---|---|
| Multi-context model warnings / large model | `context_switch_param(toposort_mode=pushdown)` or `context_switch_param(toposort_mode=dfs)` to control context split behaviour. Enum values are **lowercase strings** (`dfs`, `depthwise`, `pushdown`, `automatic`) per the DFC schema — uppercase will not parse. |
| FPS lower than expected | Try, in order: raise `performance_param(compiler_optimization_level=max)`; `allocator_param(enable_partial_row_buffers=disabled)`; `compilation_param({conv*}, mixed_mem=disabled)`; on `hailo15l` with too many LCUs, `allocator_param(enable_fixer=max_adjcents)`; if the deployment targets a specific batch, `performance_param(optimize_for_batch=X)`. Also check the runtime profiler (see step 3) for the limiting layer. |
| Power-constrained deployment | `performance_param(optimize_for_power=True)`. |

For the canonical command reference and per-symptom guidance, consult the **DFC User Guide on the Hailo Developer Zone** (https://hailo.ai/developer-zone/).

### 5. Hand-off

> Compilation complete. To run on a Hailo device, see the DFC tutorial `DFC_4_Inference_Tutorial.ipynb` (`hailo tutorial`) or use HailoRT directly.

## Notes

- The skill never modifies the DFC wheel or Model Zoo configs.
- All Model Zoo paths are relative to the Model Zoo repo root.
- Tutorial: `DFC_3_Compilation_Tutorial.ipynb` (full reference). On-device inference: `DFC_4_Inference_Tutorial.ipynb`. Extract both with `hailo tutorial`.
- The skill never edits ALLS files in `hailo_model_zoo/cfg/alls/` — any model-script changes are made on a working copy and then re-applied via `/hailo-optimize`.
