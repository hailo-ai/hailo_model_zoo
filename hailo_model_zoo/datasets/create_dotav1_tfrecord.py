import argparse
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from hailo_model_zoo.utils import downloader, path_resolver

TF_RECORD_TYPE = "val", "train"
TF_RECORD_LOC = {
    "val": "models_files/dotav1/2025-12-01/dotav1_val.tfrecord",
    "train": "models_files/dotav1/2025-12-01/dotav1_train.tfrecord",
}


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _create_tf_record(filenames, name, num_images):
    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[name])
    (tfrecords_filename.parent).mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(filenames[:num_images])
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for i, (img_path, bbox_annotations) in enumerate(progress_bar):
            progress_bar.set_description(f"{name} #{i}: {img_path}")
            x1, y1, x2, y2, x3, y3, x4, y4, category_id, is_crowd, areas = [], [], [], [], [], [], [], [], [], [], []
            img_jpeg = open(img_path, "rb").read()
            img = np.array(Image.open(img_path))
            image_height = img.shape[0]
            image_width = img.shape[1]

            for object_annotations in bbox_annotations:
                coords = object_annotations["coords"]
                category = object_annotations["category"]
                x1.append(float(coords[0]) / image_width)
                y1.append(float(coords[1]) / image_height)
                x2.append(float(coords[2]) / image_width)
                y2.append(float(coords[3]) / image_height)
                x3.append(float(coords[4]) / image_width)
                y3.append(float(coords[5]) / image_height)
                x4.append(float(coords[6]) / image_width)
                y4.append(float(coords[7]) / image_height)
                areas.append(1.0)  # dummy field for compatibility
                is_crowd.append(int(0))  # dummy field for compatibility
                category_id.append(int(category))

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "height": _int64_feature(image_height),
                        "width": _int64_feature(image_width),
                        "num_boxes": _int64_feature(len(bbox_annotations)),
                        "x1": _float_list_feature(x1),
                        "y1": _float_list_feature(y1),
                        "x2": _float_list_feature(x2),
                        "y2": _float_list_feature(y2),
                        "x3": _float_list_feature(x3),
                        "y3": _float_list_feature(y3),
                        "x4": _float_list_feature(x4),
                        "y4": _float_list_feature(y4),
                        "area": _float_list_feature(areas),
                        "is_crowd": _int64_feature(is_crowd),
                        "category_id": _int64_feature(category_id),
                        "image_name": _bytes_feature(str.encode(img_path.name)),
                        "image_jpeg": _bytes_feature(img_jpeg),
                        "image_id": _int64_feature(i),
                    }
                )
            )
            writer.write(example.SerializeToString())
    return i + 1, tfrecords_filename


def get_annotations(gt_path):
    annotations = []
    with open(gt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            coords = list(map(float, parts[1:]))
            category = int(parts[0])
            annotation = {"coords": coords, "category": category}
            annotations.append(annotation)
    return annotations


def download_dataset():
    dataset_name = "DOTAv1"
    dataset_dir = path_resolver.resolve_data_path("")
    # download images and labels if needed
    if not (dataset_dir / dataset_name / "images").is_dir() or not (dataset_dir / dataset_name / "labels").is_dir():
        print("Downloading DOTAv1 dataset...")
        filename = downloader.download_file("https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip")
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(str(dataset_dir))
        Path(filename).unlink()
    return dataset_dir / dataset_name


def run(data_dir, name, num_images):
    if data_dir == "":
        data_dir = download_dataset()
    data_dir = Path(data_dir)
    img_folder = data_dir / "images" / name
    gt_folder = data_dir / "labels" / name
    img_list = sorted([img_folder / name for name in img_folder.iterdir()])
    gt_list = sorted([gt_folder / name for name in gt_folder.iterdir() if name.suffix == ".txt"])
    gt_annotations = [get_annotations(gt_path) for gt_path in gt_list]
    img_labels_list = list(zip(img_list, gt_annotations, strict=True))
    images_num, tfrecord_name = _create_tf_record(img_labels_list, name, num_images)
    print("\nDone converting {} images".format(images_num))
    print(f"Dataset saved at {tfrecord_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", help="Which tf-record to create {}".format(TF_RECORD_TYPE))
    parser.add_argument("--data-path", help="DOTAv1 data directory", type=str, default="")
    parser.add_argument("--num-images", type=int, default=5000, help="Limit num images")
    args = parser.parse_args()
    assert args.type in TF_RECORD_TYPE, "need to provide which kind of tfrecord to create {}".format(TF_RECORD_TYPE)
    run(args.data_path, args.type, args.num_images)
