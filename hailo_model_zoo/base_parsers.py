import argparse
from pathlib import Path

from hailo_model_zoo.utils.cli_utils import OneResizeValueAction, add_model_name_arg
from hailo_model_zoo.utils.completions import (
    ALLS_COMPLETE,
    CKPT_COMPLETE,
    FILE_COMPLETE,
    HAR_COMPLETE,
    HEF_COMPLETE,
    TFRECORD_COMPLETE,
    YAML_COMPLETE,
)
from hailo_model_zoo.utils.constants import DEVICE_NAMES, TARGETS


def make_parsing_base():
    parsing_base_parser = argparse.ArgumentParser(add_help=False)
    config_group = parsing_base_parser.add_mutually_exclusive_group()
    add_model_name_arg(config_group, optional=True)
    config_group.add_argument(
        "--yaml",
        type=str,
        default=None,
        dest="yaml_path",
        help=("Path to YAML for network configuration." "By default using the default configuration"),
    ).complete = YAML_COMPLETE
    parsing_base_parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        dest="ckpt_path",
        help=("Path to onnx or ckpt to use for parsing." " By default using the model cache location"),
    ).complete = CKPT_COMPLETE
    parsing_base_parser.add_argument(
        "--hw-arch",
        type=str,
        default="hailo8",
        metavar="",
        choices=["hailo8", "hailo8l"],
        help="Which hw arch to run: hailo8 / hailo8l. By default using hailo8.",
    )
    parsing_base_parser.add_argument(
        "--start-node-names",
        type=str,
        default="",
        nargs="+",
        help="List of names of the first nodes to parse.\nExample: --start-node-names <start_name1> <start_name2> ...",
    )
    parsing_base_parser.add_argument(
        "--end-node-names",
        type=str,
        default="",
        nargs="+",
        help="List of nodes that indicate the parsing end. The order determines the order of the outputs."
        "\nExample: --end-node-names <end_name1> <end_name2> ...",
    )
    parsing_base_parser.set_defaults(results_dir=Path("./"))
    return parsing_base_parser


def make_optimization_base():
    optimization_base_parser = argparse.ArgumentParser(add_help=False)
    mutually_exclusive_group = optimization_base_parser.add_mutually_exclusive_group()
    mutually_exclusive_group.add_argument(
        "--model-script",
        type=str,
        default=None,
        dest="model_script_path",
        help="Path to model script to use. By default using the model script specified"
        " in the network YAML configuration",
    ).complete = ALLS_COMPLETE
    mutually_exclusive_group.add_argument(
        "--performance",
        action="store_true",
        help="Enable flag for benchmark performance",
    )

    optimization_base_parser.add_argument(
        "--har", type=str, default=None, help="Use external har file", dest="har_path"
    ).complete = HAR_COMPLETE
    optimization_base_parser.add_argument(
        "--calib-path",
        type=Path,
        help="Path to external tfrecord for calibration or a directory containing \
            images in jpg or png format",
    ).complete = TFRECORD_COMPLETE
    optimization_base_parser.add_argument(
        "--resize",
        type=int,
        nargs="+",
        action=OneResizeValueAction,
        help="Add input resize from given [h,w]",
    )
    optimization_base_parser.add_argument(
        "--input-conversion",
        type=str,
        choices=["nv12_to_rgb", "yuy2_to_rgb", "rgbx_to_rgb"],
        help="Add input conversion from given type",
    )
    optimization_base_parser.add_argument(
        "--classes", type=int, metavar="", help="Number of classes for NMS configuration"
    )
    return optimization_base_parser


def make_hef_base():
    hef_base_parser = argparse.ArgumentParser(add_help=False)
    hef_base_parser.add_argument(
        "--hef", type=str, default=None, help="Use external HEF files", dest="hef_path"
    ).complete = HEF_COMPLETE
    return hef_base_parser


def make_profiling_base():
    profile_base_parser = argparse.ArgumentParser(add_help=False)
    return profile_base_parser


def make_evaluation_base():
    evaluation_base_parser = argparse.ArgumentParser(add_help=False)
    targets = TARGETS
    devices = ", ".join(DEVICE_NAMES)
    evaluation_base_parser.add_argument(
        "--target",
        type=str,
        choices=targets,
        metavar="",
        default="full_precision",
        help="Which target to run: full_precision (GPU) / emulator (GPU) / hardware (PCIe).\n"
        f"A specific device may be specified. Available devices: {devices}",
    )

    evaluation_base_parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for INFERENCE (evaluation and pre-quant stats collection) only "
        "(feel free to increase to whatever your GPU can handle). "
        " the quant-aware optimizers s.a. QFT & IBC use the calibration batch size parameter from the ALLS",
    )

    evaluation_base_parser.add_argument(
        "--data-count",
        type=int,
        default=None,
        dest="eval_num_examples",
        help="Maximum number of images to use for evaluation",
    )

    evaluation_base_parser.add_argument(
        "--visualize",
        action="store_true",
        dest="visualize_results",
        help="Run visualization without evaluation. The default value is False",
    )
    evaluation_base_parser.add_argument(
        "--video-outpath",
        help="Make a video from the visualizations and save it to this path",
    ).complete = FILE_COMPLETE
    evaluation_base_parser.add_argument(
        "--data-path",
        type=Path,
        help="Path to external tfrecord for evaluation. In case you use --visualize \
            you can give a directory of images in jpg or png format",
    ).complete = TFRECORD_COMPLETE
    evaluation_base_parser.add_argument(
        "--ap-per-class",
        action="store_true",
        dest="show_results_per_class",
        help="Print AP results per class, relevant only for object detection and instance segmentation tasks",
    )

    evaluation_base_parser.add_argument(
        "--custom-infer-config",
        type=Path,
        dest="custom_infer_config",
        help="A file that indicates witch elements to set lossless or lossy",
    )
    evaluation_base_parser.set_defaults(
        print_num_examples=1e9,
        visualize_results=False,
        use_lite_inference=False,
        use_service=False,
    )
    return evaluation_base_parser
