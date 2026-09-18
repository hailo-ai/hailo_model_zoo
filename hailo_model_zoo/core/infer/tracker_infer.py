import numpy as np
from PIL import Image
from tqdm import tqdm

from hailo_model_zoo.core.factory import INFER_FACTORY
from hailo_model_zoo.core.infer.infer_utils import log_accuracy, to_numpy

TEMPLATE_FACTOR = 2.0
TEMPLATE_SIZE = 128
SEARCH_FACTOR = 5.0
SEARCH_SIZE = 320
CLIP_MARGIN = 10.0


def _sample_target(image, box, area_factor, output_sz):
    """Crop a square region around ``box`` of area ``area_factor**2`` times its own.
    Mirrors Stark's ``lib/train/data/processing_utils.sample_target``.
    Returns:
        The resized crop and the resize factor (output_sz / crop_sz), which is needed to
        map the predicted box back into image coordinates.
    """
    x, y, w, h = box
    crop_sz = int(np.ceil(np.sqrt(max(w, 1e-3) * max(h, 1e-3)) * area_factor))
    crop_sz = max(crop_sz, 1)
    x1 = int(round(x + 0.5 * w - crop_sz * 0.5))
    y1 = int(round(y + 0.5 * h - crop_sz * 0.5))
    x2, y2 = x1 + crop_sz, y1 + crop_sz
    x1_pad, y1_pad = max(0, -x1), max(0, -y1)
    x2_pad = max(x2 - image.shape[1] + 1, 0)
    y2_pad = max(y2 - image.shape[0] + 1, 0)
    patch = image[y1 + y1_pad : y2 - y2_pad, x1 + x1_pad : x2 - x2_pad, :]
    patch = np.pad(patch, ((y1_pad, y2_pad), (x1_pad, x2_pad), (0, 0)), mode="constant")
    resized = np.asarray(Image.fromarray(patch).resize((output_sz, output_sz), Image.BILINEAR))
    return resized, output_sz / crop_sz


def _clip_box(box, height, width, margin=0):
    """Port of Stark's ``lib/utils/box_ops.clip_box``.

    Kept structurally identical to the reference: corners are clamped into the frame
    (x1 to [0, W-margin], x2 to [margin, W]) and the width/height are recomputed from
    the clamped corners, rather than clamping x/y/w/h independently.
    """
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), width - margin)
    x2 = min(max(margin, x2), width)
    y1 = min(max(0, y1), height - margin)
    y2 = min(max(margin, y2), height)
    return [x1, y1, max(margin, x2 - x1), max(margin, y2 - y1)]


def _map_box_back(box_norm, state, resize_factor, height, width):
    """Turn the head's normalized xyxy (search-crop frame) into image-frame xywh."""
    box_px = np.asarray(box_norm, dtype=np.float64) * SEARCH_SIZE / resize_factor
    cx_prev = state[0] + 0.5 * state[2]
    cy_prev = state[1] + 0.5 * state[3]
    half = 0.5 * SEARCH_SIZE / resize_factor
    x1 = box_px[0] + (cx_prev - half)
    y1 = box_px[1] + (cy_prev - half)
    x2 = box_px[2] + (cx_prev - half)
    y2 = box_px[3] + (cy_prev - half)
    clipped = _clip_box([x1, y1, x2 - x1, y2 - y1], height, width, margin=CLIP_MARGIN)
    return np.asarray(clipped, dtype=np.float64)


@INFER_FACTORY.register(name="tracker_infer")
def tracker_infer(
    runner,
    context,
    logger,
    eval_num_examples,
    print_num_examples,
    batch_size,
    dataset,
    postprocessing_callback,
    eval_callback,
    visualize_callback,
    model_augmentation_callback,
    video_outpath,
    dump_results,
    results_path,
    *,
    np_infer=False,
):
    """Sequential closed-loop inference for a single-object tracker.
    The standard ``model_infer`` loop iterates a pre-built tf.data pipeline, so every
    network input has to be known before inference starts. A tracker violates that: the
    search region for frame *t* is cropped around the box predicted for frame *t-1*.
    Frames must arrive in order and grouped by ``video_name``; the LaSOT TFRecord is
    written that way. A new ``video_name`` re-initializes the tracker from that frame's
    ground-truth box, which is the standard one-shot tracking protocol.
    """
    eval_metric = eval_callback()
    if eval_num_examples:
        dataset = dataset.take(eval_num_examples)
    # Forced to 1: the loop is inherently sequential, and frames must not be reordered.
    batched_dataset = dataset.batch(1)
    network_name = runner.get_hn_model().name
    template_key = f"{network_name}/input_layer1"
    search_key = f"{network_name}/input_layer2"
    logger.info("Running tracking inference...")
    with (
        context as ctx,
        tqdm(total=None, desc="Processed", unit="frames", disable=None if not print_num_examples < 1 else True) as pbar,
    ):
        # runner.infer() rather than get_keras_model()/model(): the keras path needs
        # model.build(batched_dataset) to learn its input structure, and the crops fed
        # here are built per-frame rather than coming from the dataset pipeline.
        def predict_function(feed):
            return runner.infer(ctx, feed, batch_size=1)

        current_video = None
        template = None
        state = None
        num_frames = 0
        try:
            for frame_batch, img_info in batched_dataset:
                info = to_numpy(img_info)
                frame = to_numpy(frame_batch)[0]
                video_name = info["video_name"][0]
                video_name = video_name.decode() if isinstance(video_name, bytes) else str(video_name)
                gt_box = np.asarray(info["gt_box"][0], dtype=np.float64)
                height = int(info["height"][0])
                width = int(info["width"][0])
                if video_name != current_video:
                    # One-shot init: first frame of a sequence seeds the template from GT.
                    current_video = video_name
                    state = gt_box.copy()
                    template, _ = _sample_target(frame, state, TEMPLATE_FACTOR, TEMPLATE_SIZE)
                    pred_box = state.copy()
                else:
                    search, resize_factor = _sample_target(frame, state, SEARCH_FACTOR, SEARCH_SIZE)
                    outputs = predict_function(
                        {
                            template_key: template[None].astype(np.float32),
                            search_key: search[None].astype(np.float32),
                        }
                    )
                    # The corner-head soft-argmax runs in postprocessing, not on chip.
                    box_norm = to_numpy(postprocessing_callback(outputs)["pred_boxes"]).reshape(-1)[:4]
                    state = _map_box_back(box_norm, state, resize_factor, height, width)
                    pred_box = state.copy()
                # target_visible is carried through so the evaluator can drop frames
                # where the target is absent, matching the reference protocol. The
                # tracker still runs on them -- only the scoring skips them.
                eval_metric.update_op(
                    {"pred_boxes": pred_box[None]},
                    {
                        "gt_box": gt_box[None],
                        "video_name": [video_name],
                        "target_visible": np.asarray([int(info["target_visible"][0])]),
                    },
                )
                num_frames += 1
                pbar.update(1)
        except KeyboardInterrupt:
            pbar.close()
            logger.info("Inference interrupted by user, displaying partial results")
    eval_metric.evaluate()
    accuracy = eval_metric.get_accuracy()
    log_accuracy(logger, num_frames, accuracy)
    return accuracy
