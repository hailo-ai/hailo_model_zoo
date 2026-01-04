import argparse
from pathlib import Path

import numpy as np

try:
    import open_clip
except ModuleNotFoundError as err:
    raise ModuleNotFoundError("Module 'open_clip' not found. Please run: pip install open_clip_torch") from err

import requests
import torch
from PIL import Image
from termcolor import colored
from tfrecord import TFRecordWriter
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm

IMAGE_NAMES_URL = (
    "https://raw.githubusercontent.com/adobe-research/Cross-lingual-Test-Dataset-XTD10/main/XTD10/test_image_names.txt"
)

CAPTIONS_URL = "https://raw.githubusercontent.com/adobe-research/Cross-lingual-Test-Dataset-XTD10/main/XTD10/test_1kcaptions_en.txt"


class TFRecordWriterWrapper(TFRecordWriter):
    def __init__(self, data_path: str) -> None:
        super().__init__(data_path)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()


class CocoTextImageRetrievalDataset(Dataset):
    def __init__(self, root: str, prompts_path: str, image_file_path: str, processor, tokenizer=None) -> None:
        super().__init__()

        self.processor = processor
        self.tokenizer = tokenizer
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
        inputs = {}
        inputs["input_ids"] = self.tokenizer(text)
        inputs["pixel_values"] = self.processor(image)
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

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(args.model)
    tokenizer = open_clip.get_tokenizer(args.model)

    npz_path = f"{args.model.replace('/','-')}.npz"
    print(f"Saving model parameters to {colored(npz_path, 'cyan')}")
    with torch.no_grad():
        np.savez(
            npz_path,
            text_embeddings=model.token_embedding.weight.detach().numpy().T,
            pos_embeddings=model.positional_embedding.detach().numpy().T,
            projection_layer=model.text_projection.detach().numpy().T,
        )
    model.eval()
    model = model.to("cuda")

    dataset = CocoTextImageRetrievalDataset(
        args.root, args.prompt_file, args.image_file, processor=preprocess_train, tokenizer=tokenizer
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
    )

    print(f"Running {args.model} over entire dataset. This could take a while...")
    with torch.no_grad(), TFRecordWriterWrapper(args.outpath) as writer:
        input_ids = []
        input_embeds = []
        text_embeds = []
        image_embeds = []
        for inputs in tqdm(loader):
            for key in inputs.keys():
                inputs[key] = inputs[key].to("cuda")
            outputs = model(inputs["pixel_values"], inputs["input_ids"].squeeze(0))
            input_ids.append(inputs["input_ids"])
            image_embeds.append(outputs[0])
            input_embeds.append(model.token_embedding(inputs["input_ids"].squeeze(0)))
            text_embeds.append(outputs[1])

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
            writer.write(
                {
                    "input_ids": (to_bytes(cur_input_ids), "byte"),
                    "input_embeds": (to_bytes(cur_input_embeds), "byte"),
                    "image_embeds": (to_bytes(cur_image_embeds), "byte"),
                    "text_embeds": (to_bytes(cur_text_embeds), "byte"),
                    "input_embeds_dim": ([cur_input_embeds.shape[-1]], "int"),
                    "output_embeds_dim": ([cur_text_embeds.shape[-1]], "int"),
                }
            )


def to_bytes(tensor_or_array):
    if torch.is_tensor(tensor_or_array):
        tensor_or_array = tensor_or_array.cpu().numpy()
    return tensor_or_array.tobytes()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example: python create_coco_xtd10_resnet_tfrecord.py \
        -root /data/data/coco/images/ --prompt-file coco_prompt_file.txt --image-file coco_img_file.txt "
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model", type=str, default="hf-hub:timm/resnet50_clip.openai")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--outpath", type=str, default="coco_xtd10_en_clip_resnet50.tfrecord")
    parser.add_argument("--suffix", type=str, default="")
    main(parser.parse_args())
