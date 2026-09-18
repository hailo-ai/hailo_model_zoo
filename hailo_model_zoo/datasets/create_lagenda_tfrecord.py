#!/usr/bin/env python
r"""Create Lagenda age/gender TFRecords (calib + val) for the MiVOLO model.

Uses MiVOLO's own ReaderAgeGender to produce the exact 224x224 letterboxed face +
person crops (so the Model-Zoo pipeline matches the reference preprocessing), storing
both crops + GT (age, is_female_int) per instance.

Runs single-phase where all deps are present (TF + MiVOLO: torch/cv2/timm/ultralytics):
  python create_lagenda_tfrecord.py --split train --calib 4000 \
      --images path/to/Lagenda/images --annotations path/to/Lagenda/annotations \
      --mivolo-repo path/to/MiVOLO
  python create_lagenda_tfrecord.py --split validation --num 12000 \
      --images path/to/Lagenda/images --annotations path/to/Lagenda/annotations \
      --mivolo-repo path/to/MiVOLO

Two-phase when the crop env (MiVOLO deps) and the TFRecord env (TF) differ:
  # phase A (MiVOLO env, no TF): dump crops
  python create_lagenda_tfrecord.py --split train --calib 4000 \
      --images path/to/Lagenda/images --annotations path/to/Lagenda/annotations \
      --mivolo-repo path/to/MiVOLO --crops-out lagenda_calib_crops.pkl
  # phase B (TF env, no MiVOLO): write TFRecord
  python create_lagenda_tfrecord.py --calib 4000 --from-crops lagenda_calib_crops.pkl

With no --out the TFRecord goes to its canonical $HMZ_DATA location, which is where
cfg/base/lagenda.yaml expects it, so `hailomz optimize / eval mivolo_d1` then works
out of the box.
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

# Canonical TFRecord directory - keep in sync with cfg/base/lagenda.yaml.
# Relative to $HMZ_DATA (default ~/.hailomz/data).
TF_RECORD_LOC = "models_files/lagenda/2026-07-23"

MIVOLO_URL = "https://github.com/WildChlamydia/MiVOLO.git"

MIVOLO_MISSING_MSG = """\
Cannot import MiVOLO's ReaderAgeGender: {reason}

Crop generation (phase A) needs the MiVOLO reference code plus its dependencies
(torch / timm / ultralytics / opencv). MiVOLO pins versions that clash with the
TensorFlow stack the Model Zoo runs on, so the recommended setup is a separate
virtualenv just for this phase:

  git clone {url} path/to/MiVOLO
  python3 -m venv path/to/mivolo-venv
  source path/to/mivolo-venv/bin/activate
  pip install -r path/to/MiVOLO/requirements.txt

Then dump the crops from that env (phase A - no TensorFlow needed):

  python create_lagenda_tfrecord.py --split train --calib 4000 \\
      --images path/to/Lagenda/images --annotations path/to/Lagenda/annotations \\
      --mivolo-repo path/to/MiVOLO --crops-out lagenda_calib_crops.pkl

and convert the pickle to a TFRecord back in the Model Zoo env (phase B - no
MiVOLO needed):

  python create_lagenda_tfrecord.py --calib 4000 --from-crops lagenda_calib_crops.pkl

If MiVOLO and TensorFlow do coexist in the current environment, both phases can
run in a single command - just pass --mivolo-repo without --crops-out.
"""


def build_reader(mivolo_repo: str, images: str, annotations: str, split: str) -> Any:
    """Instantiate MiVOLO's ReaderAgeGender from a local MiVOLO clone.

    Args:
        mivolo_repo: Path to a MiVOLO checkout (prepended to ``sys.path``).
        images: Path to the Lagenda image root.
        annotations: Path to the Lagenda annotations directory.
        split: Reader split name, e.g. ``train`` or ``validation``.

    Returns:
        A ``ReaderAgeGender`` yielding ``((face_bgr, person_bgr), (age, gender))``.

    Raises:
        SystemExit: If the repo path does not exist, or MiVOLO / one of its
            dependencies is not importable - with setup instructions.
    """
    repo = Path(mivolo_repo).expanduser()
    if not (repo / "mivolo").is_dir():
        raise SystemExit(MIVOLO_MISSING_MSG.format(reason=f"no 'mivolo' package under {repo}", url=MIVOLO_URL))

    sys.path.insert(0, str(repo))
    try:
        from mivolo.data.dataset.reader_age_gender import ReaderAgeGender
    except ImportError as exc:
        raise SystemExit(MIVOLO_MISSING_MSG.format(reason=exc, url=MIVOLO_URL)) from exc

    return ReaderAgeGender(
        images_path=images,
        annotations_path=annotations,
        split=split,
        target_size=224,
        with_persons=True,
        disable_faces=False,
        only_age=False,
    )


def gen_crops(args: argparse.Namespace) -> list[dict]:
    """Phase A: MiVOLO reader -> list of {face_jpeg, person_jpeg, age, is_female, id}."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise SystemExit(MIVOLO_MISSING_MSG.format(reason=exc, url=MIVOLO_URL)) from exc

    def _jpeg(rgb):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        assert ok
        return buf.tobytes()

    reader = build_reader(args.mivolo_repo, args.images, args.annotations, args.split)
    n = len(reader)
    idx = list(range(n))
    target = args.calib if args.calib else args.num  # strided, deterministic subset
    if target and target < n:
        step = max(1, n // target)
        idx = idx[::step][:target]

    recs = []
    for i in idx:
        (face_bgr, person_bgr), (age, gender) = reader[i]
        # Skip instances missing either label - a -1 age or a non-M/F gender would
        # otherwise be written and flow unfiltered into AgeGenderEval (a -1 age skews
        # the MAE; a -1 gender is silently scored as "male"), biasing the metrics.
        if age == "-1" or gender not in ("F", "M"):
            continue
        recs.append(
            {
                "face_jpeg": _jpeg(cv2.cvtColor(np.ascontiguousarray(face_bgr), cv2.COLOR_BGR2RGB)),
                "person_jpeg": _jpeg(cv2.cvtColor(np.ascontiguousarray(person_bgr), cv2.COLOR_BGR2RGB)),
                "age": int(round(float(age))),
                "is_female": 1 if gender == "F" else 0,
                "id": i,
            }
        )
        if len(recs) % 1000 == 0:
            print(f"  crops {len(recs)}", end="\r")
    print(f"\ncollected {len(recs)} crops")
    return recs


def write_tfrecord(recs: list[dict], out: Path) -> None:
    """Phase B: list of crop dicts -> TFRecord."""
    import tensorflow as tf

    def _i64(v):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))

    def _b(v):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

    out.parent.mkdir(parents=True, exist_ok=True)
    # Context manager closes (and flushes) the writer even if a record raises, so a
    # failure never leaves a truncated TFRecord behind.
    with tf.io.TFRecordWriter(str(out)) as writer:
        for r in recs:
            ex = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "age": _i64(r["age"]),
                        "is_female_int": _i64(r["is_female"]),
                        "image_id": _i64(r["id"]),
                        "image_name": _b(str.encode(f"lagenda_{r['id']}")),
                        "face_jpeg": _b(r["face_jpeg"]),
                        "person_jpeg": _b(r["person_jpeg"]),
                    }
                )
            )
            writer.write(ex.SerializeToString())
    print(f"Done: {len(recs)} records -> {out}")


def resolve_out_dir(out: str) -> Path:
    """Return the TFRecord output directory, defaulting to the $HMZ_DATA location."""
    if out:
        return Path(out).expanduser()

    from hailo_model_zoo.utils import path_resolver

    return path_resolver.resolve_data_path(TF_RECORD_LOC)


def run(args: argparse.Namespace) -> None:
    if args.from_crops:
        with open(args.from_crops, "rb") as f:
            recs = pickle.load(f)
    else:
        recs = gen_crops(args)
        if args.crops_out:
            with open(args.crops_out, "wb") as f:
                pickle.dump(recs, f)
            print(f"crops -> {args.crops_out}")
            return
    out = resolve_out_dir(args.out) / f"lagenda_{'calib' if args.calib else 'val'}.tfrecord"
    write_tfrecord(recs, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", default="", help="phase A: path/to/Lagenda/images (image root)")
    p.add_argument("--annotations", default="", help="phase A: path/to/Lagenda/annotations (split CSVs)")
    p.add_argument("--mivolo-repo", default="", help="phase A: path/to/MiVOLO clone (added to sys.path)")
    p.add_argument("--split", default="validation", help="reader split: train or validation")
    p.add_argument("--out", default="", help=f"output directory (default: $HMZ_DATA/{TF_RECORD_LOC})")
    p.add_argument("--calib", type=int, default=0, help="if >0, calib subset of this size (also names the output)")
    p.add_argument("--num", type=int, default=0, help="if >0, cap the val set to this many instances (strided)")
    p.add_argument("--crops-out", default="", help="phase A: dump crops pickle here and stop (no TF needed)")
    p.add_argument("--from-crops", default="", help="phase B: build TFRecord from this crops pickle (no MiVOLO needed)")
    args = p.parse_args()
    if not args.from_crops:
        missing = [name for name in ("images", "annotations", "mivolo_repo") if not getattr(args, name)]
        if missing:
            p.error("crop generation requires " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))
    run(args)
