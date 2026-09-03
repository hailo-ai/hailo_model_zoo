---
name: hailo-parse
description: Parse an ONNX or TensorFlow model into a Hailo HAR using the Hailo Dataflow Compiler (DFC). Use when bringing a new model into the Hailo flow. Auto-discovers parser start/end nodes and normalization from the Model Zoo when the model name matches a known network.
argument-hint: <path-to-model> [hw_arch]
allowed-tools: Read, Write, Bash, Glob, Grep
---

# Hailo Parse

First stage of the DFC flow: take an ONNX (`.onnx`) or TensorFlow (`.tflite` / SavedModel) file and produce a Hailo Archive (`.har`) containing the network in Hailo's internal representation. The HAR is then consumed by `/hailo-optimize`.

This skill is shipped with the Hailo Model Zoo. It assumes the DFC wheel (`hailo_sdk_client`) is installed in the active virtualenv.

## Inputs

| Arg | Required | Notes |
|---|---|---|
| `<path-to-model>` | yes | Path to an `.onnx` or `.tflite` file. |
| `[hw_arch]` | no | One of `hailo10h` (default), `hailo15h`, `hailo15l` |

Optional via follow-up questions: `model_name` (defaults to filename stem), explicit `start_node_names` / `end_node_names` / `net_input_shapes` overrides.

**TF input preconditions** (verify before parsing — these are common failure modes that surface as confusing parser errors):

- **TF1 frozen graphs / checkpoints**: deprecated by the DFC since v5.1.0. Convert to TF2 SavedModel and then to TFLite (see next bullet) before parsing.
- **TF2 SavedModel**: not directly accepted by the parser. Convert to TFLite first with `tf.lite.TFLiteConverter.from_saved_model(...).convert()` and pass the resulting `.tflite`.
- **Int8-quantized TFLite**: **not supported** by the parser. Use a 32-bit or 16-bit TFLite — quantization is applied later by `/hailo-optimize`, not at parse time.

**ONNX export checklist** (PyTorch users — the most common parse-failure sources):

When the user is exporting from PyTorch with `torch.onnx.export(...)`, the following arguments are required for clean DFC parsing:

- `training=torch.onnx.TrainingMode.PRESERVE` (or `EVAL`) — otherwise BatchNorm folds incorrectly and the parsed model silently diverges from the source.
- `do_constant_folding=False`.
- `opset_version` in **15–21** (DFC supported range).
- `dynamo=False` — the new TorchDynamo exporter is not supported.

If the user can't change the export, suggest re-exporting before parsing.

## Workflow

### 1. Resolve inputs

- Verify the model file exists and detect framework (ONNX vs TF) from extension.
- Default `model_name = Path(model).stem`.
- Default `hw_arch = hailo10h`.
- Confirm `hailo_sdk_client` import works (`python -c "from hailo_sdk_client import ClientRunner"`); if not, abort with "DFC wheel not installed in this virtualenv — see https://hailo.ai/developer-zone/".

### 2. Auto-match Model Zoo precedent (silent)

Use the `hailomz` CLI to look up parser config — it resolves `base:` inheritance for you. Do **not** glob YAMLs and stitch inheritance manually.

Two options:

**a. Direct extraction with `hailomz query`** (preferred for a single field):

```bash
hailomz query 'networks[?network.network_name==`<model_name>`].parser.nodes' --cache /tmp/hailomz_networks.json
hailomz query 'networks[?network.network_name==`<model_name>`].parser.normalization_params' --cache /tmp/hailomz_networks.json
```

The `--cache` flag persists the assembled network DB so subsequent queries are fast. Use JMESPath for any field on the network config.

**b. Full resolved YAML with `hailomz cfg`** (when you need the complete config):

```bash
hailomz cfg <model_name>
```

This writes `<model_name>.yaml` to the current directory with all `base:` inheritance fully resolved.

If `<model_name>` doesn't match a known network, both commands fail cleanly — try name variants (case-insensitive, `-` ↔ `_`, with trailing version/resolution suffixes stripped, e.g. `yolov8s_640` → `yolov8s`) before falling through to step 3.

From the result, extract:

- `parser.nodes` — list of the form `[<start_node>, [<end_node_1>, <end_node_2>, ...]]`. The first element is the start node (or `null` for auto-detect). The second element is a list of end nodes; for single-output models it may be a single string instead.
- `parser.normalization_params.mean_list` and `std_list` — saved for `/hailo-optimize` to validate the calibration data later (not used at parse time).

Log the chosen model name: `Using Model Zoo network config: <model_name> (resolved via hailomz)`.

If no match, proceed to step 3.

### 3. No precedent — interactive node discovery

Tell the user how to find start/end nodes:

- Open the ONNX in **Netron** (https://netron.app) and identify the input op (typically the first `Conv` or `Gemm`) and the output op(s) **before any post-processing**.
- For YOLO families: end at the `Conv` outputs that feed NMS, **not** at the post-NMS ops. NMS will be added by `/hailo-optimize` via `nms_postprocess(...)`.
- For classifiers: usually end at the final `Softmax` or `Gemm`.

Offer to run a short ONNX I/O dump to help:

```python
import onnx
m = onnx.load("<model.onnx>")
print("inputs:", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
print("outputs:", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
```

If `translate_onnx_model` later raises an unsupported-op error, refer the user to the **DFC User Guide on the Hailo Developer Zone** (https://hailo.ai/developer-zone/) for the canonical list of supported layers. The standard workaround is to **split the graph**: set `end_node_names` just before the unsupported op, run that subgraph offline (e.g., on host CPU/GPU), and parse the rest as a separate HAR. This is how a "supported graph" that fits the DFC is built.

### 4. Generate and run the parse script

For ONNX:

```python
from hailo_sdk_client import ClientRunner

runner = ClientRunner(hw_arch="<hw_arch>")
hn, npz = runner.translate_onnx_model(
    "<model.onnx>",
    "<model_name>",
    start_node_names=[<start>],
    end_node_names=[<end>, ...],
    net_input_shapes={<start>: [1, C, H, W]},  # only when shapes are dynamic
)
runner.save_har("<model_name>.har")
```

For TensorFlow / TFLite, swap to `runner.translate_tf_model(<path>, "<model_name>", ...)`. The same `start_node_names` / `end_node_names` arguments apply.

Drop `net_input_shapes` entirely when the ONNX has fully static shapes.

Run the script with `python <script>.py` (or inline via `python -c`).

### 5. Verify

- Confirm `<model_name>.har` exists; report its size in MB.
- Suggest `hailo visualizer <model_name>.har --no-browser` to view the parsed graph as an HTML report (this is the canonical inspection tool for parsed HARs; `hailo profiler` at this stage only renders "Model overview" — its richer reports are for quantized/compiled HARs).
- **Recommended sanity check**: run the parsed model through the SDK_NATIVE emulator and compare a few outputs against the original framework. This catches start/end-node mismatches and preprocessing bugs *before* the (sometimes hours-long) optimization run blames them on quantization:
  ```python
  from hailo_sdk_client import ClientRunner, InferenceContext
  runner = ClientRunner(har="<model_name>.har")
  with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
      hailo_out = runner.infer(ctx, sample_input)
  # then compare hailo_out to the original framework's output on the same input
  ```
  Alternatively, `hailo parser --compare` (DFC CLI) does this side-by-side comparison directly against the source framework — see DFC User Guide on https://hailo.ai/developer-zone/ for usage.

### 6. Common-failure cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `UnsupportedOp` / unknown layer | Layer not in DFC supported set | Consult DFC User Guide on https://hailo.ai/developer-zone/. Split the graph: set `end_node_names` before the unsupported op, run that region offline, parse the rest separately. |
| Shape inference fails | ONNX has dynamic dims | Pass `net_input_shapes={<start>: [1, C, H, W]}` explicitly. |
| Tries to parse through NMS / TopK / detection-output | Custom postprocess in the graph | Cut at the conv/sigmoid outputs that feed it. `/hailo-optimize` will insert `nms_postprocess(...)` for you. |
| End node not found | Stale node name (model was re-exported) | Re-inspect with Netron; node names change between exports. |

### 7. Hand-off

> Parsing complete. Next: run `/hailo-optimize <model_name>.har [hw_arch]` to apply a model script and quantize.

If a Model Zoo YAML was matched in step 2, also pass the saved `mean_list` / `std_list` along to `/hailo-optimize` so it can sanity-check calibration normalization.

## Notes

- The skill never modifies the DFC wheel or Model Zoo configs — it only reads YAMLs and writes a `.har` in the current directory.
- All Model Zoo paths are relative to the Model Zoo repo root (where you ran `git clone`); run Claude Code from that directory.
- Tutorials: extract DFC tutorials with `hailo tutorial` and open `DFC_1_Parsing_Tutorial.ipynb` for the canonical reference.
