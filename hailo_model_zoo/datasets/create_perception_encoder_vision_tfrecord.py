#!/usr/bin/env python

import argparse
import json
import os
import random
import re
import zipfile
from pathlib import Path

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from tqdm import tqdm

from hailo_model_zoo.utils import downloader, path_resolver

CIFAR100_CLASSES = [
    "apple",
    "aquarium fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak tree",
    "orange",
    "orchid",
    "otter",
    "palm tree",
    "pear",
    "pickup truck",
    "pine tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow tree",
    "wolf",
    "woman",
    "worm",
]

IMAGENET_NAMES_PATH = os.path.join(os.path.dirname(__file__), "..", "core", "postprocessing", "imagenet_names.json")


def _get_imagenet_classes():
    """Load ImageNet 1000 class names, taking the first name before comma for clean prompts."""
    with open(IMAGENET_NAMES_PATH) as f:
        imagenet_names = json.load(f)
    return [imagenet_names[str(i)].split(",")[0].strip() for i in range(1, 1001)]


def _get_classes(dataset):
    if dataset == "cifar100":
        return CIFAR100_CLASSES
    elif dataset == "imagenet":
        return _get_imagenet_classes()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


DATASET_CHOICES = ["cifar100", "imagenet"]
TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
    "a blurry photo of the {}.",
    "a black and white photo of the {}.",
    "a low contrast photo of the {}.",
    "a high contrast photo of the {}.",
    "a bad photo of the {}.",
    "a good photo of the {}.",
    "a photo of the small {}.",
    "a photo of the big {}.",
]
TF_RECORD_TYPE = ["val", "calib"]
CLASS_TOKEN_LOC = {
    "PE-Core-B16-224": "models_files/cifar100/2026-03-09/class_token_pe_core_b16_224_image_encoder.npy",
    "PE-Core-L14-336": "models_files/cifar100/2026-03-09/class_token_pe_core_l14_336_image_encoder.npy",
}
MODEL_TYPE = CLASS_TOKEN_LOC.keys()
TF_RECORD_LOC = {
    "val": "models_files/cifar100/test/cifar100_val.tfrecord",
    "val_imagenet": "models_files/imagenet/2026-05-05/imagenet_clip_val.tfrecord",
    "calib": "models_files/coco/2026-09-03/coco_clip_calib2017.tfrecord",
}


def _get_class_token_path(model_name, dataset):
    """Get class token path, replacing cifar100 with imagenet in path for imagenet dataset."""
    base_path = CLASS_TOKEN_LOC[model_name]
    if dataset == "imagenet":
        return base_path.replace("cifar100", "imagenet")
    return base_path


COCO_PATH = "http://images.cocodataset.org/zips/val2017.zip"


def _generate_class_tokens(model, tokenizer, classnames, templates, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    zeroshot_weights = []
    for classname in tqdm(classnames):
        text = [template.format(classname) for template in templates]  # format with class
        text = tokenizer(text).to(device)
        with torch.no_grad():
            _, outputs, _ = model(None, text)
        class_embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
        class_embeddings = class_embeddings.mean(dim=0)
        class_embeddings = class_embeddings.detach().cpu().numpy()
        zeroshot_weights.append(class_embeddings)

    zeroshot_weights = np.stack(zeroshot_weights, axis=0)
    return zeroshot_weights


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _create_tfrecord_cifar(name, images, labels):
    """Loop over all the images in filenames and create the TFRecord"""
    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[name])
    (tfrecords_filename.parent).mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(len(images))
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for i in range(len(images)):
            progress_bar.update(1)
            img_numpy = images[i]
            height = img_numpy.shape[0]
            width = img_numpy.shape[1]
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "height": _int64_feature(height),
                        "width": _int64_feature(width),
                        "label_index": _int64_feature(labels[i][0]),
                        "image_numpy": _bytes_feature(np.array(img_numpy, np.uint8).tostring()),
                    }
                )
            )
            writer.write(example.SerializeToString())
    return i + 1


def _create_class_tokens(model_name, dataset="cifar100"):
    model = pe.CLIP.from_config(model_name, pretrained=True)
    tokenizer = transforms.get_text_tokenizer(model.context_length)
    class_token_filename = path_resolver.resolve_data_path(_get_class_token_path(model_name, dataset))
    (class_token_filename.parent).mkdir(parents=True, exist_ok=True)
    classes = _get_classes(dataset)
    class_tokens = _generate_class_tokens(model, tokenizer, classes, TEMPLATES, model_name)
    np.save(class_token_filename, class_tokens)
    print(f"Class tokens are saved in {class_token_filename}")


def _create_tfrecord_imagenet(dataset_dir):
    """Create TFRecord from ImageNet validation set directory (expects n*/ subfolders)."""
    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC["val_imagenet"])
    (tfrecords_filename.parent).mkdir(parents=True, exist_ok=True)

    lib_names = sorted(f.name for f in Path(dataset_dir).iterdir() if re.match(r"^n[0-9]+$", f.name))
    class_id = {v: i for i, v in enumerate(lib_names)}
    file_list = []
    for lib in Path(dataset_dir).iterdir():
        if lib.name not in class_id:
            continue
        for img_path in lib.iterdir():
            file_list.append((str(img_path), class_id[lib.name]))
    random.seed(0)
    random.shuffle(file_list)

    progress_bar = tqdm(file_list)
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for i, (img_path, label_index) in enumerate(progress_bar):
            progress_bar.set_description(f"val_imagenet #{i}: {Path(img_path).name}")
            img_jpeg = open(img_path, "rb").read()
            img = Image.open(img_path).convert("RGB")
            width, height = img.size
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "height": _int64_feature(height),
                        "width": _int64_feature(width),
                        "label_index": _int64_feature(label_index),
                        "image_name": _bytes_feature(str.encode(Path(img_path).name)),
                        "image_jpeg": _bytes_feature(img_jpeg),
                    }
                )
            )
            writer.write(example.SerializeToString())
    return i + 1


def run_val(type, model_name, npy_only=False, dataset="cifar100", dataset_dir=None):
    _create_class_tokens(model_name, dataset)
    if npy_only:
        return
    if dataset == "cifar100":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        images_num = _create_tfrecord_cifar(type, x_test, y_test)
    elif dataset == "imagenet":
        assert dataset_dir is not None, "Please provide --dataset-dir for ImageNet validation set"
        images_num = _create_tfrecord_imagenet(dataset_dir)
    print(f"Done converting {images_num} images")


def download_dataset(type, model_name, dataset_dir):
    filename = downloader.download_file(COCO_PATH)
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(str(dataset_dir))
    Path(filename).unlink()
    return dataset_dir


def create_calib_tfrecord(dataset_dir, type):
    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[type])
    (tfrecords_filename.parent).mkdir(parents=True, exist_ok=True)
    filenames = list(Path(dataset_dir).glob("val2017/*.jpg"))
    progress_bar = tqdm(filenames)
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for i, img_path in enumerate(progress_bar):
            progress_bar.set_description(f"{type} #{i + 1}: {img_path}")
            img = Image.open(img_path).convert("RGB")
            width, height = img.size
            np_img = np.array(img)
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "height": _int64_feature(height),
                        "width": _int64_feature(width),
                        "image_numpy": _bytes_feature(np.array(np_img, np.uint8).tostring()),
                        "label_index": _int64_feature(1),
                    }
                )
            )
            writer.write(example.SerializeToString())
            progress_bar.update(1)
    return i + 1


def run_calib(type, model_name, dataset_dir):
    if dataset_dir is None:
        dataset_dir = path_resolver.resolve_data_path("coco")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        download_dataset(type, model_name, dataset_dir)

    create_calib_tfrecord(dataset_dir, type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create TFRecord of CIFAR100 for validation and COCO for calibration to use in CLIP/Siglip\n"
            "models from HuggingFace.\n"
            "Note: If you use a GPU and encounter an Out of Memory (OOM) error, "
            "consider setting CUDA_VISIBLE_DEVICES=99 to use CPU only."
        ),
        usage='use "%(prog)s --help" for more information',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("type", help="TFRecord of which dataset to create", type=str, choices=TF_RECORD_TYPE)
    parser.add_argument(
        "--model",
        help="CLIP model to use",
        type=str,
        choices=list(CLASS_TOKEN_LOC.keys()),
    )
    parser.add_argument(
        "--dataset",
        help="Evaluation dataset to use",
        type=str,
        default="cifar100",
        choices=DATASET_CHOICES,
    )
    parser.add_argument("--npy_only", help="Create only npy for validation", action="store_true", default=False)
    parser.add_argument(
        "--dataset-dir", type=str, default=None, help="Path to dataset (ImageNet val dir or COCO for calib)"
    )

    args = parser.parse_args()
    if args.type == "val":
        run_val(args.type, args.model, args.npy_only, args.dataset, args.dataset_dir)
    elif args.type == "calib":
        run_calib(args.type, args.model, args.dataset_dir)
    else:
        raise Exception(f"TFrecord type {type} is not supported.")

"""
-----------------------------------------------------------------
CMD used to create a cifar100.tfrecord of the CIFAR100 dataset for validation:
python create_perception_encoder_tfrecord.py val --model <model_name>

If you only need the NPY files containing the text embeddings for the validation set, you can use:
python create_perception_encoder_tfrecord val --npy_only --model <model_name>

CMD used to create a coco.tfrecord of the COCO dataset for calibration (independent of the model):
python create_perception_encoder_tfrecord calib --dataset-dir <path_to_coco_dataset>
-----------------------------------------------------------------
"""
