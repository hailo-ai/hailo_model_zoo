#!/usr/bin/env python

import argparse
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from tqdm import tqdm

try:
    from transformers import AutoModel, AutoTokenizer
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "Modules AutoModel & AutoTokenizer are not installed. Please install them using pip install transformers."
    ) from err
try:
    from transformers import CLIPModel, CLIPTokenizer
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "Modules CLIPModel & CLIPTokenizer are not installed. Please install them using pip install transformers."
    ) from err

from hailo_model_zoo.utils import downloader, path_resolver

CLASSES = [
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
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K": "models_files/cifar100/test/class_token_vit_l_14_laion2B.npy",
    "openai/clip-vit-large-patch14-336": "models_files/cifar100/2025-01-13/class_token_vit_l_14_336.npy",
    "google/siglip-base-patch16-224": "models_files/cifar100/2025-03-17/class_token_siglip_base_16_224.npy",
    "google/siglip-so400m-patch14-224": "models_files/cifar100/2025-03-17/class_token_siglip_so400_16_224.npy",
    "google/siglip-large-patch16-256": "models_files/cifar100/2025-03-17/class_token_siglip_large_16_256.npy",
    "google/siglip2-base-patch16-224": "models_files/cifar100/2025-03-17/class_token_siglip2_base_16_224.npy",
    "google/siglip2-large-patch16-256": "models_files/cifar100/2025-03-17/class_token_siglip2_large_16_256.npy",
    "google/siglip2-base-patch32-256": "models_files/cifar100/2025-03-17/class_token_siglip2_base_32_256.npy",
}
TF_RECORD_LOC = {
    "val": "models_files/cifar100/test/cifar100_val.tfrecord",
    "calib": "models_files/coco/2025-07-02/coco_clip_calib2017.tfrecord",
}
PADDING_LENGTH = {
    "google/siglip-base-patch16-224": 64,
    "google/siglip-so400m-patch14-224": 16,
    "google/siglip-large-patch16-256": 64,
    "google/siglip2-base-patch16-224": 64,
    "google/siglip2-large-patch16-256": 64,
    "google/siglip2-base-patch32-256": 64,
}
COCO_PATH = "http://images.cocodataset.org/zips/val2017.zip"


def _generate_class_tokens(model, tokenizer, classnames, templates, model_name):
    zeroshot_weights = []
    for classname in tqdm(classnames):
        texts = [template.format(classname) for template in templates]  # format with class
        if model_name.startswith("google/siglip"):
            padding_length = PADDING_LENGTH[model_name]
            inputs = tokenizer(
                texts, return_tensors="pt", padding="max_length", max_length=padding_length, truncation=True
            )
        else:
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        with torch.no_grad():
            outputs = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
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


def _create_class_tokens(model_name):
    if model_name.startswith("google/siglip"):
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = CLIPModel.from_pretrained(model_name)
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
    class_token_filename = path_resolver.resolve_data_path(CLASS_TOKEN_LOC[model_name])
    (class_token_filename.parent).mkdir(parents=True, exist_ok=True)
    class_tokens = _generate_class_tokens(model, tokenizer, CLASSES, TEMPLATES, model_name)
    np.save(class_token_filename, class_tokens)
    print(f"Class tokens are saved in {class_token_filename}")


def run_val(type, model_name, npy_only=False):
    _create_class_tokens(model_name)
    if npy_only:
        return
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    images_num = _create_tfrecord_cifar(type, x_test, y_test)
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
            progress_bar.set_description(f"{type} #{i+1}: {img_path}")
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
        default="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        choices=list(CLASS_TOKEN_LOC.keys()),
    )

    parser.add_argument("--npy_only", help="Create only npy for validation", action="store_true", default=False)
    parser.add_argument("--dataset-dir", type=None, default=None, help="Path to coco for calib set")

    args = parser.parse_args()
    if args.type == "val":
        run_val(args.type, args.model, args.npy_only)
    elif args.type == "calib":
        run_calib(args.type, args.model, args.dataset_dir)
    else:
        raise Exception(f"TFrecord type {type} is not supported.")

"""
-----------------------------------------------------------------
CMD used to create a cifar100.tfrecord of the CIFAR100 dataset for validation:
python create_clip_vision_tfrecord.py val --model <model_name>

If you only need the NPY files containing the text embeddings for the validation set, you can use:
python create_clip_vision_tfrecord.py val --npy_only --model <model_name>

CMD used to create a coco.tfrecord of the COCO dataset for calibration (independent of the model):
python create_clip_vision_tfrecord.py calib --dataset-dir <path_to_coco_dataset>
-----------------------------------------------------------------
"""
