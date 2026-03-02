#!/usr/bin/env python
import argparse
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorflow as tf
from tifffile import imread
from tqdm import tqdm

random.seed(42)


def main(args):
    input_dirs = [Path(p) for p in args.images_path]
    images_list = [image_path for input_dir in input_dirs for image_path in input_dir.glob("*.tiff")]
    random.shuffle(images_list)
    images_list = images_list[: args.num_images] if args.num_images else images_list

    onnx_session = ort.InferenceSession(args.onnx_path)
    onnx_input_nodes = [node.name for node in onnx_session.get_inputs()]
    onnx_output_nodes = [node.name for node in onnx_session.get_outputs()]
    print("ONNX model loaded successfully.")

    onnx_name = Path(args.onnx_path).name.split(".")[0]
    tf_record_path = str(Path(args.output_path, onnx_name + f"_{args.type}.tfrecord"))
    print("Start creating tfrecord file at", tf_record_path)
    with tf.io.TFRecordWriter(tf_record_path) as writer:
        progress_bar = tqdm(images_list)
        for image_path in progress_bar:
            progress_bar.set_description(str(image_path))
            image = imread(
                image_path
            )  # Image from *.tiff file is 3 channels, each channel is a gray-scale image from AMAT dataset
            image_h, image_w = image.shape[1], image.shape[2]
            onnx_input_data = {
                onnx_input_nodes[0]: image[None, :1],
                onnx_input_nodes[1]: image[None, 1:2],
                onnx_input_nodes[2]: image[None, 2:],
            }
            onnx_output = onnx_session.run(onnx_output_nodes, onnx_input_data)
            onnx_output = np.squeeze(np.concatenate(onnx_output[: args.onnx_num_outputs], axis=0))

            feature_dict = {
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.transpose((1, 2, 0)).tobytes()])),
                "gt": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[onnx_output.transpose((1, 2, 0)).tobytes()])
                ),
                "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[image_h])),
                "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[image_w])),
            }

            tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            writer.write(tf_example.SerializeToString())
        print("TFRecord file created successfully at", tf_record_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python create_amat_halfunet_tfrecord.py --onnx-path halfunet_1x3x960x176.sim.onnx --output-path . "
            "--type val --images-path /data/data/datasets/AMAT/Train_Calibrate_Test_Data/960x176/test\n"
            "  python create_amat_halfunet_tfrecord.py --onnx-path halfunet_1x3x960x176.sim.onnx --output-path . "
            "--type calib --images-path /data/data/datasets/AMAT/Train_Calibrate_Test_Data/960x176/train"
            " /data/data/datasets/AMAT/Train_Calibrate_Test_Data/960x176/train_planted\n"
        ),
    )
    parser.add_argument(
        "--images-path",
        help="Paths to AMAT images (*tiff format), can be multiple paths",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        required=True,
        help="ONNX path for generating ground truth",
    )
    parser.add_argument("--num-images", type=int, default=None, help="Limit num images")
    parser.add_argument("--onnx-num-outputs", type=int, default=3, help="Use 4 outputs from onnx or less")
    parser.add_argument("--output-path", type=Path, required=True, help="Output tfrecord file path")
    parser.add_argument("--type", type=str, default="val", choices=["calib", "val"], help="Type of dataset to create")

    args = parser.parse_args()
    main(args)
