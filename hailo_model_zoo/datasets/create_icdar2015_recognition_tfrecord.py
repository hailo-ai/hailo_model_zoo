#!/usr/bin/env python

import argparse
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from hailo_model_zoo.utils import downloader, path_resolver

TF_RECORD_TYPE = "val", "calib"
TF_RECORD_LOC = {
    "val": "models_files/icdar2015/2025-08-30/icdar15_text_rec_val_set.tfrecord",
    "calib": "models_files/icdar2015/2025-08-30/icdar15_text_rec_calib_set.tfrecord",
}

ICDAR2015_PATH = "https://www.kaggle.com/api/v1/datasets/download/bestofbests9/icdar2015"


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def crop_text_region(box, image, max_aspect_ratio=320 / 48):
    """
    Crop the text region from the image based on the bounding box.
    Args:
        box: Bounding box coordinates in format [x1, y1, x2, y2, x3, y3, x4, y4]
        image: Input image as numpy array
        max_aspect_ratio: Maximum allowed width/height ratio for the cropped region

    Returns:
        numpy.ndarray: Cropped image region, or None if aspect ratio exceeds threshold

    Raises:
        ValueError: If box format is invalid or image is empty
    """

    if box is None or len(box) != 8:
        raise ValueError("Box must contain exactly 8 coordinates [x1,y1,x2,y2,x3,y3,x4,y4]")

    if image is None or image.size == 0:
        raise ValueError("Image cannot be None or empty")

    # Reshape box coordinates to 4x2 array of points
    pts = box.reshape(4, 2).astype("uint16")

    # Find bounding rectangle coordinates
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]

    left = max(0, np.min(x_coords))
    right = min(image.shape[1], np.max(x_coords))
    top = max(0, np.min(y_coords))
    bottom = min(image.shape[0], np.max(y_coords))

    width = right - left
    height = bottom - top
    aspect_ratio = width / height

    # Check aspect ratio constraint
    # Recognition model allows padding on width so width/height is not allowed to exceed required max_aspect_ratio
    if aspect_ratio > max_aspect_ratio:
        return None

    # Crop the image region
    cropped_image = image[top:bottom, left:right]

    return cropped_image


def get_anns(gt_path, image):
    """
    Extract annotations and crop text regions from an image based on ground truth file.

    Args:
        gt_path: Path to the ground truth annotation file
        image: Input image as numpy array

    Returns:
        dict: Dictionary containing 'cropped_text' (list of cropped images) and 'tags' (list of text labels)

    Raises:
        FileNotFoundError: If ground truth file doesn't exist
        ValueError: If annotation format is invalid
    """
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    with open(gt_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    tags = []
    cropped_text = []
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:  # Skip empty lines
            continue

        # Parse annotation line
        annotation_parts = line.split(",")
        if len(annotation_parts) != 9:
            print(f"Warning: Skipping invalid annotation on line {line_idx + 1}: insufficient data")
            continue
        annotation_parts = annotation_parts[:9]

        # Skip ignore boxes marked with "###"
        text_label = annotation_parts[-1]
        if text_label == "###":
            continue

        # Parse bounding box coordinates
        coords = [int(annotation_parts[i]) for i in range(8)]
        pts = np.array(coords, dtype=np.float32)

        # Crop text region from image
        cropped_region = crop_text_region(pts, image)
        if cropped_region is None:
            continue  # Skip if cropping failed
        cropped_text.append(cropped_region)
        tags.append(text_label)

    return {
        "cropped_text": cropped_text,
        "tags": tags,
    }


def create_tfrecord(dataset_dir, dataset_type, num_images):
    """
    Create TFRecord file from ICDAR2015 dataset images and annotations.
    Args:
        dataset_dir: Path to the dataset directory
        dataset_type: Type of dataset ("calib" or "val")
        num_images: Maximum number of text regions to process

    Returns:
        int: Total number of text regions processed and written to TFRecord
    """

    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[dataset_type])
    if tfrecords_filename.exists():
        print(f"tfrecord already exists at {tfrecords_filename}. Skipping...")
        return 0
    tfrecords_filename.parent.mkdir(parents=True, exist_ok=True)

    # Get dataset directories
    images_dir, annotations_dir = get_img_anns_dirs(dataset_dir, dataset_type)
    image_files = list(images_dir.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"No .jpg files found in {images_dir}")

    print(f"Found {len(image_files)} images in {images_dir}")
    print(f"Creating {dataset_type} TFRecord: {tfrecords_filename}")

    # Process images and create TFRecord
    total_text_regions = 0
    processed_images = 0

    progress_bar = tqdm(image_files, desc=f"Processing {dataset_type}")
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for img_path in progress_bar:
            if total_text_regions >= num_images:
                break
            progress_bar.set_description(
                f"{dataset_type} | Regions: {total_text_regions}/{num_images} | Image: {img_path.name}"
            )

            img = cv2.imread(str(img_path))

            # Get corresponding annotation file
            ann_name = "gt_" + img_path.stem.split(".")[0] + ".txt"
            ann_path = annotations_dir / ann_name

            annotations = get_anns(ann_path, img)
            for cropped_text, tag in zip(annotations["cropped_text"], annotations["tags"]):
                if total_text_regions >= num_images:
                    break

                height, width, _ = cropped_text.shape

                # Encode image as JPEG
                success, encoded_img = cv2.imencode(".jpg", cropped_text)
                if not success:
                    continue  # Skip if encoding failed
                img_jpeg = encoded_img.tobytes()

                # assumes filename format like "img_123.jpg"
                image_id = int(img_path.stem.split("_")[1])
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "height": _int64_feature(height),
                            "width": _int64_feature(width),
                            "text": _bytes_feature(str.encode(tag)),
                            "cropped_text": _bytes_feature(img_jpeg),
                            "image_name": _bytes_feature(str.encode(img_path.name)),
                            "image_id": _int64_feature(image_id),
                            "text_id": _int64_feature(total_text_regions + 1),
                        }
                    )
                )

                writer.write(example.SerializeToString())
                total_text_regions += 1
            processed_images += 1

    print("\nProcessing complete:")
    print(f"  - Total text regions: {total_text_regions}")
    print(f"  - Images processed: {processed_images}")
    print(f"  - TFRecord saved: {tfrecords_filename}")

    return total_text_regions


def get_img_anns_dirs(dataset_dir, dataset_type):
    key = "training" if dataset_type == "calib" else "test"
    images_dir = dataset_dir / f"ch4_{key}_images"
    annotations_dir = dataset_dir / f"ch4_{key}_localization_transcription_gt"
    return images_dir, annotations_dir


def download_dataset(dataset_dir):
    if dataset_dir is None or dataset_dir == "":
        dataset_dir = Path.cwd()
        dataset_dir = dataset_dir / "icdar2015_raw"
    else:
        dataset_dir = Path(dataset_dir)

    if dataset_dir.is_dir():
        print("ICDAR2015 dataset already exists at {}".format(dataset_dir))
    else:
        print("Downloading ICDAR2015 dataset to {}".format(dataset_dir))
        with tempfile.NamedTemporaryFile() as outfile:
            downloader.download_to_file(ICDAR2015_PATH, outfile)
            with zipfile.ZipFile(outfile, "r") as zip_ref:
                zip_ref.extractall(str(dataset_dir))

    return dataset_dir


def run(dataset_dir, dataset_type, num_images):
    dataset_dir = download_dataset(dataset_dir)
    images_num = create_tfrecord(dataset_dir, dataset_type, num_images)
    if images_num > 0:
        print(f"Dataset saved at {TF_RECORD_LOC[dataset_type]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", help="which tf-record to create {}".format(TF_RECORD_TYPE), choices=["calib", "val"])
    parser.add_argument(
        "--dataset_dir", type=str, default="icdar2015_raw", help="Path to ICDAR2015 dataset base directory"
    )
    parser.add_argument("--num-images", type=int, default=8192, help="Limit num images")

    args = parser.parse_args()
    run(args.dataset_dir, args.type, args.num_images)
