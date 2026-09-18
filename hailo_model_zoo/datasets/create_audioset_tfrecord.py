#!/usr/bin/env python

"""Build AudioSet calibration + evaluation TFRecords for YamNet.

Three modes (pick one):

  --synthetic                 generate calib+eval from synthesized waveforms
                              (tones / chirps / noise / silence). Useful for
                              CI / smoke tests when AudioSet is unavailable.

  --hf-dataset NAME           stream real audio from a HuggingFace dataset
                              (default: agkphysics/AudioSet) that re-hosts the
                              AudioSet FLAC clips. No YouTube downloads. Uses
                              the vendored core/postprocessing/yamnet_class_map.json
                              to map AudioSet MIDs -> YamNet's 521 indices unless
                              overridden by --class-map.

  --audio-dir DIR --csv CSV   convert local 16 kHz mono WAV files referenced by
                              an AudioSet segments CSV (eval_segments.csv or
                              balanced_train_segments.csv). The CSV has the
                              upstream format: YTID,start_seconds,end_seconds,
                              positive_labels (`/m/...` MIDs comma-separated).
                              Requires --class-map.

Each TFRecord entry is one (96, 64, 1) log-mel patch + the clip-level
multi-hot label (521-d, YamNet's subset of the AudioSet ontology).

Audio preprocessing (16 kHz / 25 ms window / 10 ms hop / Hann / 64 mel
bands 125-7500 Hz / log+0.001 / 96-frame patches at 0.48 s hop) follows
Google's reference YamNet implementation. Parameters validated against an
independent numpy reimplementation within float tolerance.

You can produce calib and eval in one invocation, or run twice with
different `--hf-calib-config` / `--hf-eval-config` to control which HF
split is used for which side.
"""

import argparse
import csv
import io
import json
from math import gcd
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.signal import resample_poly

from hailo_model_zoo.utils import path_resolver

# Canonical TFRecord locations — keep in sync with cfg/base/audioset.yaml.
# Paths are relative to $HMZ_DATA (default ~/.hailomz).
TF_RECORD_LOC = {
    "calib": "models_files/audioset/2026-06-01/audioset_calib.tfrecord",
    "eval": "models_files/audioset/2026-06-01/audioset_eval.tfrecord",
}

# --- YamNet feature parameters (DO NOT CHANGE — bound by trained weights) ----
SAMPLE_RATE = 16000
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
MEL_BANDS = 64
MEL_MIN_HZ = 125.0
MEL_MAX_HZ = 7500.0
LOG_OFFSET = 0.001
PATCH_WINDOW_SECONDS = 0.96
PATCH_HOP_SECONDS = 0.48
NUM_YAMNET_CLASSES = 521

WINDOW_LENGTH_SAMPLES = int(round(SAMPLE_RATE * STFT_WINDOW_LENGTH_SECONDS))  # 400
HOP_LENGTH_SAMPLES = int(round(SAMPLE_RATE * STFT_HOP_LENGTH_SECONDS))  # 160
FFT_LENGTH = 512
PATCH_FRAMES = int(round(PATCH_WINDOW_SECONDS / STFT_HOP_LENGTH_SECONDS))  # 96
PATCH_HOP_FRAMES = int(round(PATCH_HOP_SECONDS / STFT_HOP_LENGTH_SECONDS))  # 48


# --- Feature extraction ------------------------------------------------------


def _waveform_to_log_mel_patches(waveform):
    waveform = tf.cast(waveform, tf.float32)
    stft = tf.signal.stft(
        waveform,
        frame_length=WINDOW_LENGTH_SAMPLES,
        frame_step=HOP_LENGTH_SAMPLES,
        fft_length=FFT_LENGTH,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )
    magnitude = tf.abs(stft)
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=MEL_BANDS,
        num_spectrogram_bins=magnitude.shape[-1],
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=MEL_MIN_HZ,
        upper_edge_hertz=MEL_MAX_HZ,
    )
    log_mel = tf.math.log(tf.matmul(magnitude, mel_weights) + LOG_OFFSET)
    return tf.signal.frame(log_mel, frame_length=PATCH_FRAMES, frame_step=PATCH_HOP_FRAMES, axis=0).numpy()


# --- TFRecord schema ---------------------------------------------------------


def _bytes_feature(v):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def _int64_feature(v):
    if isinstance(v, (int, np.integer)):
        v = [v]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))


def _patch_to_example(patch, label_multi_hot, clip_id, patch_index):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "patch_bytes": _bytes_feature(patch.astype(np.float32).tobytes()),
                "patch_height": _int64_feature(PATCH_FRAMES),
                "patch_width": _int64_feature(MEL_BANDS),
                "label_multi_hot_bytes": _bytes_feature(label_multi_hot.astype(np.int8).tobytes()),
                "num_classes": _int64_feature(NUM_YAMNET_CLASSES),
                "clip_id": _bytes_feature(clip_id.encode("utf-8")),
                "patch_index": _int64_feature(patch_index),
            }
        )
    )


# --- Synthetic generator (CI / smoke test path) ------------------------------


def _synthetic_clips(rng, count, duration_s=10.0):
    n = int(duration_s * SAMPLE_RATE)
    kinds = ["tone", "chirp", "noise", "silence"]
    for i in range(count):
        kind = kinds[i % len(kinds)]
        t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
        if kind == "tone":
            freq = float(rng.uniform(200, 4000))
            wave = 0.3 * np.sin(2 * np.pi * freq * t)
        elif kind == "chirp":
            f0 = float(rng.uniform(200, 1000))
            f1 = float(rng.uniform(2000, 6000))
            wave = 0.3 * np.sin(2 * np.pi * (f0 + (f1 - f0) * t / duration_s) * t)
        elif kind == "noise":
            wave = 0.1 * rng.standard_normal(n).astype(np.float32)
        else:
            wave = (1e-4 * rng.standard_normal(n)).astype(np.float32)
        num_labels = int(rng.integers(1, 4))
        labels = rng.choice(NUM_YAMNET_CLASSES, size=num_labels, replace=False)
        yield wave.astype(np.float32), labels, f"synth_{kind}_{i:05d}"


# --- Real AudioSet path (CSV-driven) -----------------------------------------


def _load_class_map(class_map_path):
    """Load the YamNet class map.

    Accepts either:
      * `.json` (preferred / vendored): {"<index>": {"mid": "/m/...", "display_name": "..."}}
      * `.csv` (legacy): 3 cols `index,mid,display_name`

    Returns dict {mid: yamnet_index}. The original AudioSet ontology has 632
    classes but YamNet outputs only 521 — the class map encodes that subset.
    """
    class_map_path = Path(class_map_path)
    mid_to_index = {}
    if class_map_path.suffix.lower() == ".json":
        with open(class_map_path) as f:
            data = json.load(f)
        for index_str, entry in data.items():
            mid_to_index[entry["mid"]] = int(index_str)
    else:
        with open(class_map_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            cols = {c.lower(): i for i, c in enumerate(header)}
            idx_col = cols.get("index", 0)
            mid_col = cols.get("mid", 1)
            for row in reader:
                mid_to_index[row[mid_col]] = int(row[idx_col])
    if len(mid_to_index) != NUM_YAMNET_CLASSES:
        raise ValueError(f"class map has {len(mid_to_index)} entries, expected {NUM_YAMNET_CLASSES}")
    return mid_to_index


def _read_wav_16k_mono(wav_path):
    raw = tf.io.read_file(str(wav_path))
    audio, sr = tf.audio.decode_wav(raw, desired_channels=1)
    if int(sr) != SAMPLE_RATE:
        raise ValueError(f"{wav_path}: expected {SAMPLE_RATE} Hz, got {int(sr)}")
    return tf.squeeze(audio, axis=-1).numpy()


def _real_clips(audio_dir, csv_path, class_map_csv):
    mid_to_idx = _load_class_map(class_map_csv)
    audio_dir = Path(audio_dir)
    with open(csv_path) as f:
        # AudioSet CSVs have 3 lines of comment header starting with '#'
        rows = [line for line in f if not line.startswith("#") and line.strip()]
    for line in rows:
        parts = next(csv.reader([line]))
        if len(parts) < 4:
            # Malformed row: AudioSet CSVs always have 4 fields
            # (YTID, start_seconds, end_seconds, positive_labels)
            print(f"  skipping malformed CSV row: {line.strip()!r}")
            continue
        yt_id = parts[0]
        positive_mids = [m.strip() for m in parts[3].split(",")]
        wav = audio_dir / f"{yt_id}.wav"
        if not wav.exists():
            # Best-effort: skip clips we couldn't fetch
            continue
        try:
            wave = _read_wav_16k_mono(wav)
        except ValueError as e:
            print(f"  skipping {wav.name}: {e}")
            continue
        labels = np.array(
            [mid_to_idx[m] for m in positive_mids if m in mid_to_idx],
            dtype=np.int64,
        )
        if labels.size == 0:
            continue  # all positives are AudioSet-only classes, not in YamNet's 521
        yield wave, labels, yt_id


# --- HuggingFace path (agkphysics/AudioSet or any HF dataset with same schema) -


# Default location of the vendored YamNet class map (sibling-tree of this file).
_VENDORED_CLASS_MAP = Path(__file__).resolve().parent.parent / "core" / "postprocessing" / "yamnet_class_map.json"


def _resample_to_16k(wave, src_sr):
    """Resample a 1-D float waveform to 16 kHz mono via scipy polyphase.

    The agkphysics/AudioSet mirror is 48 kHz (most clips) or 44.1 kHz
    (~10 %). We reduce by gcd to choose an efficient up/down factor pair.
    """
    if src_sr == SAMPLE_RATE:
        return wave.astype(np.float32)
    g = gcd(int(src_sr), SAMPLE_RATE)
    up = SAMPLE_RATE // g
    down = int(src_sr) // g
    return resample_poly(wave, up, down).astype(np.float32)


def _hf_clips(dataset_name, config, split, class_map_csv, take, hf_cache_dir=None):
    """Stream `take` clips from a HuggingFace AudioSet-shaped dataset.

    Yields (wave_16k_mono, label_indices_in_yamnet_space, clip_id).
    """
    import soundfile as sf
    from datasets import Audio, load_dataset

    mid_to_idx = _load_class_map(class_map_csv)

    print(f"  streaming HF: {dataset_name} config={config} split={split}")
    kwargs = {"streaming": True}
    if config is not None:
        kwargs["name"] = config
    if hf_cache_dir is not None:
        kwargs["cache_dir"] = str(hf_cache_dir)
    ds = load_dataset(dataset_name, split=split, **kwargs)
    ds = ds.cast_column("audio", Audio(decode=False))

    yielded = 0
    skipped_no_labels = 0
    for ex in ds:
        if take is not None and yielded >= take:
            break
        audio = ex["audio"]
        raw = audio.get("bytes")
        if raw is None:
            # Some HF mirrors store paths instead of bytes; we don't currently
            # support that mode because it requires downloading every file.
            continue
        with io.BytesIO(raw) as f:
            try:
                wave, sr = sf.read(f, dtype="float32", always_2d=False)
            except Exception as e:  # pragma: no cover (robustness for bad clips)
                print(f"   skipping {ex.get('video_id')}: decode failed ({e})")
                continue
        if wave.ndim == 2:
            wave = wave.mean(axis=1)
        wave = _resample_to_16k(wave, sr)

        positive_mids = ex.get("labels") or []
        if isinstance(positive_mids, str):
            positive_mids = [positive_mids]
        labels_idx = np.array(
            [mid_to_idx[m] for m in positive_mids if m in mid_to_idx],
            dtype=np.int64,
        )
        if labels_idx.size == 0:
            skipped_no_labels += 1
            continue
        # video_id is the canonical AudioSet YouTube ID and is always set on
        # the agkphysics mirror; fall back to the FLAC file stem (not the
        # full path) for any non-standard mirror.
        clip_id = ex.get("video_id")
        if not clip_id:
            p = audio.get("path") or ""
            clip_id = Path(p).stem or f"hf_{yielded}"
        yield wave, labels_idx, clip_id
        yielded += 1

    if skipped_no_labels:
        print(f"   ({skipped_no_labels} clips skipped: labels not in YamNet's 521-class space)")


# --- Writer driver -----------------------------------------------------------


def _write_clips(clip_iter, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_clips = 0
    n_patches = 0
    with tf.io.TFRecordWriter(str(out_path)) as w:
        for wave, label_idx, clip_id in clip_iter:
            patches = _waveform_to_log_mel_patches(tf.constant(wave))
            multi_hot = np.zeros(NUM_YAMNET_CLASSES, dtype=np.int8)
            multi_hot[label_idx] = 1
            for i, patch in enumerate(patches):
                w.write(_patch_to_example(patch[..., None], multi_hot, clip_id, i).SerializeToString())
                n_patches += 1
            n_clips += 1
    print(f"  wrote {n_patches} patches from {n_clips} clips -> {out_path}")


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--synthetic", action="store_true", help="Use synthetic waveforms (CI / smoke test path).")
    mode.add_argument(
        "--hf-dataset",
        nargs="?",
        const="agkphysics/AudioSet",
        default=None,
        help="Stream from HuggingFace (default: agkphysics/AudioSet). "
        "Required: --out-calib and/or --out-eval. AudioSet redistribution remains "
        "the user's responsibility; review the dataset's license before use.",
    )
    p.add_argument(
        "--hf-calib-config",
        default="balanced",
        help="HF config for the calib split (default: balanced).",
    )
    p.add_argument(
        "--hf-calib-split",
        default="train",
        help="HF split for the calib data (default: train).",
    )
    p.add_argument(
        "--hf-eval-config",
        default="balanced",
        help="HF config for the eval split (default: balanced).",
    )
    p.add_argument(
        "--hf-eval-split",
        default="test",
        help="HF split for the eval data (default: test).",
    )
    p.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface).",
    )
    p.add_argument(
        "--audio-dir", type=Path, help="Local AudioSet path: directory of 16 kHz mono WAVs (filenames `<yt_id>.wav`)."
    )
    p.add_argument("--csv", dest="csv_path", type=Path, help="Local AudioSet path: e.g. eval_segments.csv.")
    p.add_argument(
        "--class-map",
        type=Path,
        default=None,
        help="yamnet_class_map.json (preferred) or .csv. Defaults to the vendored copy under core/postprocessing/.",
    )
    p.add_argument(
        "--out-calib",
        type=Path,
        default=None,
        help=f"Output path for the calibration TFRecord (default: $HMZ_DATA/{TF_RECORD_LOC['calib']}).",
    )
    p.add_argument(
        "--out-eval",
        type=Path,
        default=None,
        help=f"Output path for the evaluation TFRecord (default: $HMZ_DATA/{TF_RECORD_LOC['eval']}).",
    )
    p.add_argument("--calib-clips", type=int, default=1024, help="Number of calib clips (~19 patches each).")
    p.add_argument("--eval-clips", type=int, default=512, help="Number of eval clips.")
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = _build_arg_parser().parse_args()

    # Default behaviour: if no output paths given, write both to canonical
    # $HMZ_DATA locations matching cfg/base/audioset.yaml. Explicit --out-*
    # narrows the run to that side only.
    if args.out_calib is None and args.out_eval is None:
        args.out_calib = path_resolver.resolve_data_path(TF_RECORD_LOC["calib"])
        args.out_eval = path_resolver.resolve_data_path(TF_RECORD_LOC["eval"])

    class_map = args.class_map or _VENDORED_CLASS_MAP
    if not (args.synthetic or args.hf_dataset is not None) and not class_map.exists():
        raise SystemExit(f"--class-map not found at {class_map}")

    if args.synthetic:
        rng_calib = np.random.default_rng(args.seed)
        if args.out_calib is not None:
            print("=== Writing synthetic calibration TFRecord ===")
            _write_clips(_synthetic_clips(rng_calib, args.calib_clips), args.out_calib)
        if args.out_eval is not None:
            rng_eval = np.random.default_rng(args.seed + 1)
            print("=== Writing synthetic eval TFRecord ===")
            _write_clips(_synthetic_clips(rng_eval, args.eval_clips), args.out_eval)
        return

    if args.hf_dataset is not None:
        if args.out_calib is not None:
            print(f"=== Writing calibration TFRecord from HF ({args.hf_dataset}) ===")
            _write_clips(
                _hf_clips(
                    args.hf_dataset,
                    args.hf_calib_config,
                    args.hf_calib_split,
                    class_map,
                    take=args.calib_clips,
                    hf_cache_dir=args.hf_cache_dir,
                ),
                args.out_calib,
            )
        if args.out_eval is not None:
            print(f"=== Writing eval TFRecord from HF ({args.hf_dataset}) ===")
            _write_clips(
                _hf_clips(
                    args.hf_dataset,
                    args.hf_eval_config,
                    args.hf_eval_split,
                    class_map,
                    take=args.eval_clips,
                    hf_cache_dir=args.hf_cache_dir,
                ),
                args.out_eval,
            )
        return

    if not (args.audio_dir and args.csv_path):
        raise SystemExit("Pick a source: --synthetic, --hf-dataset, or --audio-dir + --csv.")

    print(f"=== Building TFRecord from local AudioSet ({args.csv_path.name}) ===")
    clips = list(_real_clips(args.audio_dir, args.csv_path, class_map))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(clips)
    n_calib = args.calib_clips if args.out_calib else 0
    if args.out_calib:
        _write_clips(iter(clips[:n_calib]), args.out_calib)
    if args.out_eval:
        _write_clips(iter(clips[n_calib : n_calib + args.eval_clips]), args.out_eval)


if __name__ == "__main__":
    main()
