import argparse
from pathlib import Path

import numpy as np
import requests
import tensorflow as tf

try:
    import torch
except ImportError as e:
    raise ImportError(
        "We suggest installing the following version (compatible with CUDA 11.8):\n"
        "  pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 "
        "--index-url https://download.pytorch.org/whl/cu118"
    ) from e
from PIL import Image
from termcolor import colored
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor, CLIPModel

from hailo_model_zoo.utils import path_resolver

IMAGE_NAMES_URL = (
    "https://raw.githubusercontent.com/adobe-research/Cross-lingual-Test-Dataset-XTD10/main/XTD10/test_image_names.txt"
)

CAPTIONS_URL = "https://raw.githubusercontent.com/adobe-research/Cross-lingual-Test-Dataset-XTD10/main/XTD10/test_1kcaptions_en.txt"


PADDING_LENGTH = {
    "google/siglip-base-patch16-224": 64,
    "google/siglip-so400m-patch14-224": 16,
    "google/siglip-large-patch16-256": 64,
    "google/siglip2-base-patch16-224": 64,
    "google/siglip2-large-patch16-256": 64,
    "google/siglip2-base-patch32-256": 64,
}

TF_RECORD_LOC = {
    "google/siglip-base-patch16-224": (
        "models_files/ZeroShotClassification/siglip/siglip_base_patch16_224"
        "/coco_10xtd/2025-04-06/coco_xtd10_en_siglip_base_patch16_224.tfrecord"
    ),
    "google/siglip-large-patch16-256": (
        "models_files/ZeroShotClassification/siglip/siglip_large_patch16_256"
        "/coco_10xtd/2025-04-06/coco_xtd10_en_siglip_large_patch16_256.tfrecord"
    ),
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K": (
        "models_files/clip/vit_l_14_laion2B/pretrained/2024-09-24/coco_xtd10_en_clip_vit_l_14_laion2B.tfrecord"
    ),
    "openai/clip-vit-large-patch14": (
        "models_files/clip/vit_large/coco_10xtd/2024-08-25/coco_xtd10_en_clip_vit_large.tfrecord"
    ),
    "openai/clip-vit-base-patch16": (
        "models_files/clip/vitb_16/coco_10xtd/2024-11-13/coco_xtd10_en_clip_vitb_16.tfrecord"
    ),
    "openai/clip-vit-base-patch32": (
        "models_files/clip/vitb_32/coco_10xtd/2024-11-13/coco_xtd10_en_clip_vitb_32.tfrecord"
    ),
    "google/siglip2-base-patch16-224": (
        "models_files/ZeroShotClassification/siglip/siglip2_base_patch16_224"
        "/coco_10xtd/2025-05-12/coco_xtd10_en_siglip2_base_patch16_224.tfrecord"
    ),
    "google/siglip2-large-patch16-256": (
        "models_files/ZeroShotClassification/siglip/siglip2_large_patch16_256"
        "/coco_10xtd/2025-05-12/coco_xtd10_en_siglip2_large_patch16_256.tfrecord"
    ),
    "google/siglip2-base-patch32-256": (
        "models_files/ZeroShotClassification/siglip/siglip2_base_patch32_256"
        "/coco_10xtd/2025-05-21/coco_xtd10_en_siglip2_base_patch32_256.tfrecord"
    ),
    "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M": (
        "models_files/ZeroShotClassification/clip/tinyclip/tinyclip_vit_61m_32_text_29m_laion400m_text_encoder/coco_10xtd/2025-07-21/coco_xtd10_en_vit_61m_32.tfrecord"
    ),
    "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M": (
        "models_files/ZeroShotClassification/clip/tinyclip/tinyclip_vit_40m_32_text_19m_laion400m_text_encoder/coco_10xtd/2025-07-21/coco_xtd10_en_vit_40m_32.tfrecord"
    ),
    "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M": (
        "models_files/ZeroShotClassification/clip/tinyclip/tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder/coco_10xtd/2025-07-21/coco_xtd10_en_vit_39m_16.tfrecord"
    ),
    "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M": (
        "models_files/ZeroShotClassification/clip/tinyclip/tinyclip_vit_8m_16_text_3m_yfcc15m_text_encoder/coco_10xtd/2025-07-21/coco_xtd10_en_vit_8m_16.tfrecord"
    ),
}

NPZ_RECORD_LOC = {
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K": (
        "models_files/clip/vit_l_14_laion2B/pretrained/2024-09-24/hg-clip-vit-l-14-laion2b.npz"
    ),
    "openai/clip-vit-large-patch14": (
        "models_files/clip/vit_large/coco_10xtd/2024-08-25/openai-clip-vit-large-patch14.npz"
    ),
    "openai/clip-vit-base-patch16": (
        "models_files/clip/vitb_16/pretrained/2024-11-13/openai-clip-vit-base-patch16.npz"
    ),
    "openai/clip-vit-base-patch32": (
        "models_files/clip/vitb_32/pretrained/2024-11-13/openai-clip-vit-base-patch32.npz"
    ),
    "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M": (
        "models_files/ZeroShotClassification/clip/tinyclip/tinyclip_vit_61m_32_text_29m_laion400m_text_encoder/pretrained/2025-07-21/TinyCLIP-ViT-61M-32-Text-29M-LAION400M.npz"
    ),
    "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M": (
        "models_files/ZeroShotClassification/clip/tinyclip/tinyclip_vit_40m_32_text_19m_laion400m_text_encoder/pretrained/2025-07-21/TinyCLIP-ViT-61M-32-Text-29M-LAION400M.npz"
    ),
    "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M": (
        "models_files/ZeroShotClassification/clip/tinyclip/tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder/pretrained/2025-07-21/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M.npz"
    ),
    "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M": (
        "models_files/ZeroShotClassification/clip/tinyclip/tinyclip_vit_8m_16_text_3m_yfcc15m_text_encoder/pretrained/2025-07-21/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.npz"
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
    def __init__(self, root: str, prompts_path: str, image_file_path: str, model_name: str, processor) -> None:
        super().__init__()

        self.max_length = PADDING_LENGTH.get(model_name, 77)
        self.processor = processor
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
        image = self.images[index]
        image = Image.open(image)
        text = self.prompts[index]
        inputs = self.processor(
            text=text, images=image, return_tensors="pt", padding="max_length", max_length=self.max_length
        )
        return inputs


def main(args):
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

    if args.model.startswith("google/siglip"):
        model = AutoModel.from_pretrained(args.model)
    else:
        model = CLIPModel.from_pretrained(args.model)

    # save npz only for openai/clip models. For siglip we output the single token embedding
    if (
        args.model.startswith("openai/clip-")
        or args.model.startswith("laion/CLIP-")
        or args.model.startswith("wkcn/TinyCLIP-")
    ):
        npz_filename = path_resolver.resolve_data_path(NPZ_RECORD_LOC[args.model])
        (npz_filename.parent).mkdir(parents=True, exist_ok=True)

        print(f"Saving model parameters to {colored(npz_filename, 'cyan')}")
        with torch.no_grad():
            np.savez(
                npz_filename,
                text_embeddings=model.text_model.embeddings.token_embedding.weight.numpy(),
                pos_embeddings=model.text_model.embeddings.position_embedding.weight.numpy(),
                projection_layer=model.text_projection.weight.numpy(),
            )

    model.eval()
    if torch.cuda.is_available():
        print(f"Using GPU: {colored(torch.cuda.get_device_name(0), 'green')}")
    else:
        print("Using CPU")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model)

    dataset = CocoTextImageRetrievalDataset(
        args.root, args.prompt_file, args.image_file, args.model, processor=processor
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
    )

    print(f"Running {args.model} over entire dataset. This could take a while...")
    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[args.model])
    (tfrecords_filename.parent).mkdir(parents=True, exist_ok=True)

    with torch.no_grad(), TFRecordWriterWrapper(str(tfrecords_filename)) as writer:
        input_ids = []
        input_embeds = []
        text_embeds = []
        image_embeds = []
        text_pooler_output = []
        last_text_hidden_state = []
        for inputs in tqdm(loader):
            inputs = inputs.to(model.device)
            outputs = model(input_ids=inputs.input_ids, pixel_values=inputs.pixel_values.squeeze(0))
            input_ids.append(inputs.input_ids)
            image_embeds.append(outputs.image_embeds)
            input_embeds.append(model.text_model.embeddings.token_embedding(inputs.input_ids))
            text_embeds.append(outputs.text_embeds)
            text_pooler_output.append(outputs.text_model_output.pooler_output)
            last_text_hidden_state.append(outputs.text_model_output.last_hidden_state)

        text_embeds = torch.concatenate(text_embeds, axis=0)
        image_embeds = torch.concatenate(image_embeds, axis=0)

        # normalized features - no need to normalize since the model normalizes for us
        # image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        # text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

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
        ) in tqdm(zip(input_ids, input_embeds, image_embeds, text_embeds)):
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
        description="Example: python create_coco_xtd10_vit_tfrecord.py \
        --root /data/data/coco/images/ --prompt-file coco_prompt_file.txt --image-file coco_img_file.txt "
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--outpath", type=str, default="coco_xtd10_en_clip.tfrecord")
    parser.add_argument("--suffix", type=str, default="")
    main(parser.parse_args())
