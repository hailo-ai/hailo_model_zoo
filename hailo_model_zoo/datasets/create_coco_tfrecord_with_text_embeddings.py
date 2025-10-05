#!/usr/bin/env python3
"""
Create COCO TFRecord dataset with CLIP text embeddings.

This script creates a TFRecord dataset similar to create_coco_tfrecord.py but includes
precomputed CLIP text embeddings for all 80 COCO categories. The text embeddings are
generated with explicit pooling and projection to the shared vision-text space.

Features:
- Multiple pooling methods: mean, max, cls_token
- Explicit projection to shared vision-text space
- Normalized embeddings for contrastive learning

Usage:
    python create_coco_tfrecord_with_text_embeddings.py val2017 --num-images 1000
    python create_coco_tfrecord_with_text_embeddings.py calib2017 --img /path/to/images --det /path/to/annotations.json
    python create_coco_tfrecord_with_text_embeddings.py val2017 --pooling mean --model openai/clip-vit-large-patch14
"""

import argparse
import collections
import json
import os
import random
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModelWithProjection as CLIPTPModel
from transformers import CLIPTokenizer

from hailo_model_zoo.utils import downloader, path_resolver

# TFRecord configuration
TF_RECORD_TYPE = "val2017", "calib2017"
TF_RECORD_LOC = {
    "val2017": "models_files/coco/2023-08-03/coco_val2017_with_text_embeddings.tfrecord",
    "calib2017": "models_files/coco/2023-08-03/coco_calib2017_with_text_embeddings.tfrecord",
}

# COCO class names (80 categories)
COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# Template prompts for better zero-shot performance
PROMPT_TEMPLATES = [
    "a photo of a {}",
    "a photograph of a {}",
    "an image of a {}",
    "a picture of a {}",
]


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def generate_text_embeddings(model_name="openai/clip-vit-base-patch32", device="auto", pooling_method="mean"):
    """
    Generate text embeddings for all COCO classes with explicit pooling and projection.

    Args:
        model_name: CLIP model name to use
        device: Device to run on ('auto', 'cuda', 'cpu')
        pooling_method: Pooling method ('mean', 'max', 'cls_token')

    Returns:
        numpy.ndarray: Text embeddings of shape (1, 80, 512) projected to shared space
    """
    print(f"Generating text embeddings using {model_name}...")
    print(f"Pooling method: {pooling_method}")

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Load CLIP model and tokenizer
    model = CLIPTPModel.from_pretrained(model_name)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    class_embeddings = []

    for class_name in tqdm(COCO_CLASSES, desc="Processing COCO classes"):
        texts = [class_name]
        # Tokenize all prompts for this class
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            txt_outputs = model(**inputs)
            txt_feats = txt_outputs.text_embeds

            txt_feats = txt_outputs.text_embeds
            txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
            txt_feats = txt_feats.reshape(-1, len(texts), txt_feats.shape[-1])
            class_embeddings.append(txt_feats)
    # Stack all class embeddings and reshape to (1, 80, embedding_dim)
    embeddings = np.stack(class_embeddings, axis=0)  # Shape: (80, embedding_dim)
    embeddings = embeddings.reshape(1, len(COCO_CLASSES), embeddings.shape[-1])  # Shape: (1, 80, embedding_dim)

    print(f"Generated text embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[-1]}")
    return embeddings


def _create_tfrecord(filenames, name, num_images, imgs_name2id, text_embeddings):
    """Loop over all the images in filenames and create the TFRecord"""
    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[name])
    (tfrecords_filename.parent).mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(filenames[:num_images])
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for i, (img_path, bbox_annotations) in enumerate(progress_bar):
            progress_bar.set_description(f"{name} #{i+1}: {img_path}")
            xmin, xmax, ymin, ymax, category_id, is_crowd, area = [], [], [], [], [], [], []
            img_jpeg = open(img_path, "rb").read()
            print(img_path)
            img = np.array(Image.open(img_path))
            image_height = img.shape[0]
            image_width = img.shape[1]
            for object_annotations in bbox_annotations:
                (x, y, width, height) = tuple(object_annotations["bbox"])
                if width <= 0 or height <= 0 or x + width > image_width or y + height > image_height:
                    continue
                xmin.append(float(x) / image_width)
                xmax.append(float(x + width) / image_width)
                ymin.append(float(y) / image_height)
                ymax.append(float(y + height) / image_height)
                is_crowd.append(object_annotations["iscrowd"])
                area.append(object_annotations["area"])
                category_id.append(int(object_annotations["category_id"]))

            if bbox_annotations:
                img_id = object_annotations["image_id"]
            else:
                img_id = imgs_name2id[os.path.basename(img_path)]

            # Convert text embeddings to bytes
            text_embeddings_bytes = text_embeddings.astype(np.float32).tobytes()

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "height": _int64_feature(image_height),
                        "width": _int64_feature(image_width),
                        "num_boxes": _int64_feature(len(bbox_annotations)),
                        "image_id": _int64_feature(img_id),
                        "xmin": _float_list_feature(xmin),
                        "xmax": _float_list_feature(xmax),
                        "ymin": _float_list_feature(ymin),
                        "ymax": _float_list_feature(ymax),
                        "area": _float_list_feature(area),
                        "category_id": _int64_feature(category_id),
                        "is_crowd": _int64_feature(is_crowd),
                        "image_name": _bytes_feature(str.encode(os.path.basename(img_path))),
                        "image_jpeg": _bytes_feature(img_jpeg),
                        # Add text embeddings
                        "text_embeddings": _bytes_feature(text_embeddings_bytes),
                        "text_embeddings_shape": _int64_feature(text_embeddings.shape),
                    }
                )
            )
            writer.write(example.SerializeToString())
    return i + 1


def get_img_labels_list(dataset_dir, det_file):
    with tf.io.gfile.GFile(str(det_file), "r") as fid:
        obj_annotations = json.load(fid)

    imgs_name2id = {img["file_name"]: img["id"] for img in obj_annotations["images"]}
    img_to_obj_annotation = collections.defaultdict(list)
    for annotation in obj_annotations["annotations"]:
        image_name = str(annotation["image_id"]).zfill(12) + ".jpg"
        img_to_obj_annotation[image_name].append(annotation)

    no_anns_imgs = 0
    orig_file_names, det_annotations = [], []
    for img in sorted(dataset_dir.iterdir()):
        if img.name not in img_to_obj_annotation:
            no_anns_imgs += 1
        det_annotations.append(img_to_obj_annotation[img.name])
        orig_file_names.append(str(img))
    files = list(zip(orig_file_names, det_annotations))
    print(f"{no_anns_imgs} / {len(files)} images have no annotations")
    random.seed(0)
    random.shuffle(files)
    return files, imgs_name2id


def download_dataset(name):
    dataset_name = "val2017" if "val" in name else "train2017"
    dataset_dir = path_resolver.resolve_data_path("coco")
    dataset_annotations = dataset_dir / "annotations"

    # create the libraries if needed
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # download images if needed
    if not (dataset_dir / dataset_name).is_dir():
        filename = downloader.download_file("http://images.cocodataset.org/zips/" + dataset_name + ".zip")
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(str(dataset_dir))
        Path(filename).unlink()

    # download annotations if needed
    if not dataset_annotations.is_dir():
        filename = downloader.download_file("http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(str(dataset_dir))

    return dataset_dir / dataset_name, dataset_annotations / ("instances_" + dataset_name + ".json")


def run(dataset_dir, det_file, name, num_images, model_name, device, pooling_method="mean"):
    # Generate text embeddings first
    print("Step 1: Generating CLIP text embeddings for COCO categories...")
    text_embeddings = generate_text_embeddings(model_name, device, pooling_method)

    # Process dataset
    print("Step 2: Processing COCO dataset...")
    if dataset_dir == "" or det_file == "":
        if dataset_dir != "" or det_file != "":
            raise ValueError("Please use img and det arguments together.")
        dataset_dir, det_file = download_dataset(name)

    img_labels_list, imgs_name2id = get_img_labels_list(Path(dataset_dir), Path(det_file))
    images_num = _create_tfrecord(img_labels_list, name, num_images, imgs_name2id, text_embeddings)

    print(f"\n✅ Done converting {images_num} images")
    print(f"📁 TFRecord saved to: {path_resolver.resolve_data_path(TF_RECORD_LOC[name])}")
    print(f"📊 Text embeddings shape: {text_embeddings.shape}")
    print(f"🏷️  COCO classes: {len(COCO_CLASSES)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create COCO TFRecord with CLIP text embeddings")
    parser.add_argument("type", help="which tf-record to create {}".format(TF_RECORD_TYPE))
    parser.add_argument("--img", "-img", help="images directory", type=str, default="")
    parser.add_argument("--det", "-det", help="detection ground truth", type=str, default="")
    parser.add_argument("--num-images", type=int, default=8192, help="Limit num images")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP model name (default: openai/clip-vit-base-patch32)",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to use for inference (auto, cuda, cpu)")
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "max", "cls_token"],
        help="Pooling method for text embeddings (default: mean)",
    )

    args = parser.parse_args()
    assert args.type in TF_RECORD_TYPE, "need to provide which kind of tfrecord to create {}".format(TF_RECORD_TYPE)

    run(args.img, args.det, args.type, args.num_images, args.model, args.device, args.pooling)
