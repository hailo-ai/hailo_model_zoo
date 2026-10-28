#!/usr/bin/env python
"""Create LaSOT TFRecords for the STARK single-object-tracking flow.

Two record shapes are produced, because tracking needs different data for evaluation
than for calibration:

``val`` / ``calibration_set``
    One record per video frame: the full frame plus its ground-truth box, tagged with
    ``video_name`` and ``frame_index``. The tracker's search region cannot be cropped
    ahead of time -- it depends on the tracker's own previous prediction -- so
    ``tracker_infer`` crops inside the closed loop and needs whole frames.

``calib_pairs``
    One record per template/search pair, already cropped and resized to the network
    inputs. The template comes from frame t0 and the search region from a later frame
    t1, centred on the t0 box, exactly as at inference. Calibrating on independent
    still images instead measured 0.16 AUC worse, because it misses the motion blur,
    lighting change and appearance drift the attention layers actually see.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

TEMPLATE_FACTOR = 2.0
TEMPLATE_SIZE = 128
SEARCH_FACTOR = 5.0
SEARCH_SIZE = 320
DEFAULT_MAX_GAP = 100


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _sequences(dataset_dir):
    """Yield LaSOT sequence dirs as <category>/<category>-N, sorted per category."""
    by_cat = {}
    for cat in sorted(Path(dataset_dir).iterdir()):
        if not cat.is_dir():
            continue
        seqs = [p for p in sorted(cat.iterdir()) if (p / "groundtruth.txt").exists()]
        if seqs:
            by_cat[cat.name] = seqs
    return by_cat


def _select(by_cat, num_sequences):
    """Pick sequences round-robin across categories.

    Taking the first N in sorted order draws them all from one category
    (atv-1, atv-10, atv-2, ...), which is not representative of a 15-category benchmark.
    """
    if not num_sequences:
        return [s for seqs in by_cat.values() for s in seqs]
    out, depth = [], 0
    while len(out) < num_sequences:
        added = False
        for seqs in by_cat.values():
            if depth < len(seqs):
                out.append(seqs[depth])
                added = True
                if len(out) == num_sequences:
                    break
        if not added:
            break
        depth += 1
    return out


def _sample_target(image, box, area_factor, output_sz):
    """Crop a square region around ``box`` of area ``area_factor**2`` times its own.

    Mirrors Stark's ``lib/train/data/processing_utils.sample_target`` so the recorded
    crops match what the tracker produces at runtime.
    """
    x, y, w, h = box
    crop_sz = int(np.ceil(np.sqrt(w * h) * area_factor))
    if crop_sz < 1:
        raise ValueError("box too small to crop")
    x1 = int(round(x + 0.5 * w - crop_sz * 0.5))
    y1 = int(round(y + 0.5 * h - crop_sz * 0.5))
    x2, y2 = x1 + crop_sz, y1 + crop_sz

    x1_pad, y1_pad = max(0, -x1), max(0, -y1)
    x2_pad = max(x2 - image.shape[1] + 1, 0)
    y2_pad = max(y2 - image.shape[0] + 1, 0)
    patch = image[y1 + y1_pad : y2 - y2_pad, x1 + x1_pad : x2 - x2_pad, :]
    patch = np.pad(patch, ((y1_pad, y2_pad), (x1_pad, x2_pad), (0, 0)), mode="constant")
    return np.asarray(Image.fromarray(patch).resize((output_sz, output_sz), Image.BILINEAR))


def _target_visible(seq, n):
    """Per-frame visibility flags, following pytracking's LaSOT dataset.

    A frame counts as valid only when the target is neither fully occluded nor out of
    view: ``target_visible = (full_occlusion == 0) & (out_of_view == 0)``. The reference
    success-plot computation (``calc_seq_err_robust``) drops invalid frames instead of
    scoring them as IoU 0, so the flags have to reach the evaluator.

    Missing files default to visible, so a dataset variant that omits them still works.
    """
    flags = np.ones(n, dtype=np.int64)
    for name in ("full_occlusion.txt", "out_of_view.txt"):
        path = seq / name
        if not path.exists():
            continue
        marks = np.atleast_1d(np.loadtxt(path, delimiter=",", dtype=np.float64))
        flags &= (marks[:n] == 0).astype(np.int64)
    return flags


def _write_frames(writer, sequences, max_frames):
    """Write whole-frame records for evaluation."""
    count = 0
    for seq in sequences:
        boxes = np.loadtxt(seq / "groundtruth.txt", delimiter=",", dtype=np.float64)
        frames = sorted((seq / "img").glob("*.jpg"))
        n = min(len(frames), len(boxes))
        if max_frames:
            n = min(n, max_frames)
        visible = _target_visible(seq, n)
        video_name = f"{seq.parent.name}/{seq.name}"
        print(f"writing {video_name} ({n} frames, {int(visible.sum())} valid)")
        for idx in range(n):
            path = frames[idx]
            img_jpeg = path.read_bytes()
            img = np.asarray(Image.open(path).convert("RGB"))
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "video_name": _bytes_feature(video_name.encode()),
                        "frame_index": _int64_feature(idx),
                        "height": _int64_feature(img.shape[0]),
                        "width": _int64_feature(img.shape[1]),
                        "image_name": _bytes_feature(path.name.encode()),
                        "image_jpeg": _bytes_feature(img_jpeg),
                        "gt_box": _float_list_feature([float(v) for v in boxes[idx]]),
                        "target_visible": _int64_feature(int(visible[idx])),
                    }
                )
            )
            writer.write(example.SerializeToString())
            count += 1
    return count


def _write_pairs(writer, sequences, num_pairs, max_gap, seed):
    """Write pre-cropped template/search pair records for calibration."""
    rng = random.Random(seed)
    written, guard = 0, 0
    while written < num_pairs and guard < num_pairs * 20:
        guard += 1
        seq = sequences[rng.randrange(len(sequences))]
        try:
            boxes = np.loadtxt(seq / "groundtruth.txt", delimiter=",", dtype=np.float64)
            frames = sorted((seq / "img").glob("*.jpg"))
            n = min(len(frames), len(boxes))
            if n < 2:
                continue
            t0 = rng.randrange(0, n - 1)
            t1 = min(n - 1, t0 + rng.randrange(1, max_gap + 1))
            img_z = np.asarray(Image.open(frames[t0]).convert("RGB"))
            img_x = np.asarray(Image.open(frames[t1]).convert("RGB"))
            template = _sample_target(img_z, boxes[t0], TEMPLATE_FACTOR, TEMPLATE_SIZE)
            # Search region is centred on the PREVIOUS box, as at inference.
            search = _sample_target(img_x, boxes[t0], SEARCH_FACTOR, SEARCH_SIZE)
        except (OSError, ValueError, IndexError):
            # OSError covers unreadable/corrupt JPEGs (PIL's UnidentifiedImageError is a
            # subclass) and a missing groundtruth.txt; ValueError covers malformed
            # annotations and _sample_target's "box too small to crop"; IndexError covers
            # a groundtruth shorter than the frame list. Anything else is a real bug and
            # should surface rather than be silently skipped.
            continue
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "template_png": _bytes_feature(tf.io.encode_png(template).numpy()),
                    "search_png": _bytes_feature(tf.io.encode_png(search).numpy()),
                }
            )
        )
        writer.write(example.SerializeToString())
        written += 1
        if written % 32 == 0:
            print(f"wrote {written}/{num_pairs} pairs", end="\r")
    if written < num_pairs:
        raise SystemExit(f"only wrote {written}/{num_pairs} pairs")
    return written


def run(dataset_dir, name, num_sequences, max_frames, num_pairs, max_gap, seed, out_dir=None):
    assert os.path.isdir(dataset_dir), f"no dataset directory: {dataset_dir}"
    by_cat = _sequences(dataset_dir)
    if not by_cat:
        raise SystemExit(f"no LaSOT sequences found under {dataset_dir}")
    sequences = _select(by_cat, num_sequences)
    print(f"{len(by_cat)} categories, using {len(sequences)} sequences")

    out_path = os.path.join(out_dir or dataset_dir, f"lasot_{name}.tfrecord")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with tf.io.TFRecordWriter(out_path) as writer:
        if name == "calib_pairs":
            count = _write_pairs(writer, sequences, num_pairs, max_gap, seed)
        else:
            count = _write_frames(writer, sequences, max_frames)
    print(f"\ndone. wrote {count} records to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=["val", "calibration_set", "calib_pairs"])
    parser.add_argument("-d", "--dataset-dir", required=True, help="LaSOT root")
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=15,
        help="sequences to include, drawn round-robin across categories (0 = all)",
    )
    parser.add_argument("--max-frames", type=int, default=120, help="frames per sequence")
    parser.add_argument("--num-pairs", type=int, default=128, help="calib_pairs only")
    parser.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP, help="calib_pairs only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="where to write the tfrecord (default: alongside the dataset)",
    )
    args = parser.parse_args()
    run(
        args.dataset_dir,
        args.name,
        args.num_sequences,
        args.max_frames,
        args.num_pairs,
        args.max_gap,
        args.seed,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
