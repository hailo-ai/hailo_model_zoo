import argparse

import torch
import torchvision
from diffusers.models import AutoencoderKL, AutoencoderTiny
from tfrecord import TFRecordWriter
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.datasets.coco import CocoDetection
from tqdm.auto import tqdm


class TFRecordWriterWrapper(TFRecordWriter):
    def __init__(self, data_path: str) -> None:
        super().__init__(data_path)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()


def main(args):
    model = args.model
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(model, subfolder=args.subfolder)
    vae.to("cuda")
    decoder = vae

    if args.tiny:
        decoder = AutoencoderTiny.from_pretrained(args.tiny_model)
        decoder.to(vae.device)

    scale = vae.scaling_factor / decoder.scaling_factor

    batch_size = args.batch_size
    size = args.size
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((size, size)),
            torchvision.transforms.PILToTensor(),
        ]
    )
    root = args.root
    annotation_file = args.ann
    dataset = CocoDetection(
        root=root,
        annFile=annotation_file,
        transform=transforms,
    )

    if args.num_samples:
        indices = range(args.num_samples)
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: torch.stack(tuple(b[0] for b in batch)),
    )

    psnr = PeakSignalNoiseRatio().to(vae.device)
    ssim = StructuralSimilarityIndexMeasure().to(vae.device)

    generator = torch.Generator(device=vae.device).manual_seed(args.seed)
    with torch.no_grad(), TFRecordWriterWrapper(f"sd1.5_vae_coco_{size}{args.suffix}.tfrecord") as writer:
        for images in tqdm(loader):
            target_images = images.to(vae.device, vae.dtype)
            target_images /= 255.0
            encoded = vae.encode(target_images).latent_dist
            encoded = encoded.sample(generator=generator)
            for enc, image in zip(encoded, images):
                writer.write(
                    {
                        "encoded_image": (enc.permute(1, 2, 0).cpu().numpy().tobytes(), "byte"),
                        "gt_image": (image.permute(1, 2, 0).cpu().numpy().tobytes(), "byte"),
                        "height": (size, "int"),
                        "width": (size, "int"),
                    }
                )

            encoded *= scale
            result = decoder.decode(encoded, return_dict=False)[0]
            psnr.update(preds=result, target=target_images)
            ssim.update(preds=result, target=target_images)

    print(f"PSNR: {psnr.compute()}")
    print(f"SSIM: {ssim.compute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--subfolder", type=str, default="vae")
    parser.add_argument("--no-subfolder", action="store_const", const=None, dest="subfolder")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-samples", "-n", type=int, default=None)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--ann", type=str, required=True)
    parser.add_argument("--tiny", action="store_true", default=False)
    parser.add_argument("--tiny-model", type=str, default="madebyollin/taesd")

    parser.add_argument("--suffix", type=str, default="")
    main(parser.parse_args())
