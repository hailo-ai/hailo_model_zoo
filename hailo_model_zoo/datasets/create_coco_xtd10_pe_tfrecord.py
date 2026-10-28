import argparse
import sys
from pathlib import Path

import numpy as np
import requests

try:
    import torch
except ImportError as e:
    raise ImportError(
        "We suggest installing the following version (compatible with CUDA 11.8):\n"
        "  pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 "
        "--index-url https://download.pytorch.org/whl/cu118"
    ) from e
import tensorflow as tf
from PIL import Image
from termcolor import colored
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm

from hailo_model_zoo.utils import path_resolver

IMAGE_NAMES_URL = (
    "https://raw.githubusercontent.com/adobe-research/Cross-lingual-Test-Dataset-XTD10/main/XTD10/test_image_names.txt"
)

CAPTIONS_URL = "https://raw.githubusercontent.com/adobe-research/Cross-lingual-Test-Dataset-XTD10/main/XTD10/test_1kcaptions_en.txt"

SUPPORTED_MODELS = ("PE-Core-B16-224", "PE-Core-L14-336")

TF_RECORD_LOC = {
    "PE-Core-B16-224": (
        "models_files/ZeroShotClassification/pe_core/pe_core_b16_224"
        "/coco_10xtd/2026-03-16/coco_xtd10_en_pe_core_b16_224.tfrecord"
    ),
    "PE-Core-L14-336": (
        "models_files/ZeroShotClassification/pe_core/pe_core_l14_336"
        "/coco_10xtd/2026-03-16/coco_xtd10_en_pe_core_l14_336.tfrecord"
    ),
}

NPZ_RECORD_LOC = {
    "PE-Core-B16-224": (
        "models_files/ZeroShotClassification/pe_core/pe_core_b16_224/pretrained/2026-03-16/pe_core_b16_224.npz"
    ),
    "PE-Core-L14-336": (
        "models_files/ZeroShotClassification/pe_core/pe_core_l14_336/pretrained/2026-03-16/pe_core_l14_336.npz"
    ),
}


class TFRecordWriterWrapper:
    def __init__(self, data_path: str) -> None:
        self.writer = tf.io.TFRecordWriter(data_path)

    def write(self, example: tf.train.Example) -> None:
        self.writer.write(example)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.writer.close()


class CocoTextImageRetrievalDataset(Dataset):
    def __init__(self, root: str, prompts_path: str, image_file_path: str, tokenizer, image_transform) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.image_transform = image_transform

        with open(prompts_path) as prompt_file:
            self.prompts = [line.strip() for line in prompt_file.readlines()]

        with open(image_file_path) as image_file:
            image_paths = [line.strip() for line in image_file.readlines()]

        self.images = []
        root = Path(root)
        for path in image_paths:
            path = path.replace("COCO_", "")
            full_path = root / path.replace("2014_", "2017/")
            if full_path.exists():
                self.images.append(full_path)
                continue
            full_path = str(full_path)
            if "train" in full_path:
                full_path = full_path.replace("train", "val")
            else:
                full_path = full_path.replace("val", "train")

            full_path = Path(full_path)
            if not full_path.exists():
                raise ValueError(f"Missing {path}")

            self.images.append(full_path)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        pixel_values = self.image_transform(image)
        text = self.prompts[index]
        input_ids = self.tokenizer(text).squeeze(0)
        input_ids[input_ids == 0] = 49407  # pad with eos token
        return input_ids, pixel_values


def main(args):
    sys.path.insert(0, args.pe_path)
    from core.vision_encoder.pe import CLIP
    from core.vision_encoder.transforms import get_image_transform, get_text_tokenizer

    if args.model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {args.model}. Choose from {SUPPORTED_MODELS}")

    if not Path(args.image_file).is_file():
        print(f"{colored(args.image_file, 'cyan')} not found. Downloading...")
        response = requests.get(IMAGE_NAMES_URL)
        with open(args.image_file, "wb") as file:
            file.write(response.content)

    if not Path(args.prompt_file).is_file():
        print(f"{colored(args.prompt_file, 'cyan')} not found. Downloading...")
        response = requests.get(CAPTIONS_URL)
        with open(args.prompt_file, "wb") as file:
            file.write(response.content)

    model = CLIP.from_config(args.model, pretrained=True)

    # Save NPZ with text encoder parameters
    npz_filename = path_resolver.resolve_data_path(NPZ_RECORD_LOC[args.model])
    npz_filename.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving model parameters to {colored(npz_filename, 'cyan')}")
    with torch.no_grad():
        npz_data = {
            "text_embeddings": model.token_embedding.weight.numpy(),
            "pos_embeddings": model.positional_embedding.numpy(),
        }
        if isinstance(model.text_projection, torch.nn.Linear):
            npz_data["projection_layer"] = model.text_projection.weight.numpy().T
        else:
            npz_data["projection_layer"] = model.text_projection.numpy().T
        np.savez(npz_filename, **npz_data)

    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"Using GPU: {colored(torch.cuda.get_device_name(0), 'green')}")
    else:
        print("Using CPU")
    model.to(device)

    tokenizer = get_text_tokenizer(model.context_length)
    image_transform = get_image_transform(model.image_size)

    dataset = CocoTextImageRetrievalDataset(
        args.root, args.prompt_file, args.image_file, tokenizer=tokenizer, image_transform=image_transform
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)

    print(f"Running {args.model} over entire dataset. This could take a while...")
    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[args.model])
    tfrecords_filename.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad(), TFRecordWriterWrapper(str(tfrecords_filename)) as writer:
        input_ids = []
        input_embeds = []
        text_embeds = []
        image_embeds = []
        for batch_input_ids, batch_pixel_values in tqdm(loader):
            batch_input_ids = batch_input_ids.to(device)
            batch_pixel_values = batch_pixel_values.to(device)

            image_features, text_features, _ = model(image=batch_pixel_values, text=batch_input_ids)

            input_ids.append(batch_input_ids)
            image_embeds.append(image_features)
            input_embeds.append(model.token_embedding(batch_input_ids))
            text_embeds.append(text_features)

        text_embeds = torch.concatenate(text_embeds, axis=0)
        image_embeds = torch.concatenate(image_embeds, axis=0)

        # cosine similarity as logits
        logit_scale = model.logit_scale.exp()
        cosine_similarity_matrix = torch.matmul(text_embeds, image_embeds.t())
        cosine_similarity = torch.mean(torch.diag(cosine_similarity_matrix)).item()
        logits_per_text = cosine_similarity_matrix * logit_scale
        probs = logits_per_text.softmax(dim=1)
        target = torch.eye(1000)
        retrieval_at_10 = torch.gather(target, dim=-1, index=probs.topk(10, dim=-1).indices.cpu()).sum() / target.sum()
        retrieval_at_10 = round(retrieval_at_10.item(), 4)
        print(f"Retrieval@10: {colored(retrieval_at_10, 'green')}")
        print(f"cosine similarity: {colored(f'{cosine_similarity:.3f}', 'green')}")
        print("Writing tfrecord...")
        for (
            cur_input_ids,
            cur_input_embeds,
            cur_image_embeds,
            cur_text_embeds,
        ) in tqdm(zip(input_ids, input_embeds, image_embeds, text_embeds, strict=True)):
            cur_input_embeds = cur_input_embeds.squeeze()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "input_ids": tf.train.Feature(bytes_list=tf.train.BytesList(value=[to_bytes(cur_input_ids)])),
                        "input_embeds": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[to_bytes(cur_input_embeds)])
                        ),
                        "image_embeds": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[to_bytes(cur_image_embeds)])
                        ),
                        "text_embeds": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[to_bytes(cur_text_embeds)])
                        ),
                        "input_embeds_dim": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[cur_input_embeds.shape[-1]])
                        ),
                        "output_embeds_dim": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[cur_text_embeds.shape[-1]])
                        ),
                    }
                )
            )

            writer.write(example.SerializeToString())


def to_bytes(tensor_or_array):
    if torch.is_tensor(tensor_or_array):
        tensor_or_array = tensor_or_array.cpu().numpy()
    return tensor_or_array.tobytes()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create COCO-XTD10 TFRecord with PE-Core text/image embeddings.\n"
        "Example: python create_coco_xtd10_pe_tfrecord.py "
        "--model PE-Core-B16-224 --root /data/coco/images/ --prompt-file captions.txt --image-file images.txt"
    )
    parser.add_argument("--pe-path", type=str, default="/workspace/perception_models")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--image-file", type=str, required=True)
    main(parser.parse_args())
