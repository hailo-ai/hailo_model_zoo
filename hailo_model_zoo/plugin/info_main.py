import argparse

from hailo_model_zoo.core.info_utils import get_network_info
from hailo_model_zoo.utils.cli_utils import add_model_name_arg, register_command
from hailo_model_zoo.utils.logger import get_logger


def info_model(model_name, network_info, logger):
    def build_dict(info):
        keys_list = [
            "task",
            "input_shape",
            "output_shape",
            "operations",
            "parameters",
            "framework",
            "training_data",
            "validation_data",
            "eval_metric",
            "full_precision_result",
            "source",
            "license_url",
        ]
        info_vals = [info[key_curr] for key_curr in keys_list]
        info_dict = dict(zip(keys_list, info_vals))
        return keys_list, info_dict

    keys_list, info_dict = build_dict(network_info["info"])
    msgs_list = []
    for key_curr in keys_list:
        msg = "\t{0:<25}{1}".format(key_curr + ":", info_dict[key_curr])
        msgs_list.append(msg)
    msg_w_line = "\033[0m\n" + "\n".join(msgs_list)
    logger.info(msg_w_line)


def make_info_base():
    info_base_parser = argparse.ArgumentParser(add_help=False, description="Print model information.")
    add_model_name_arg(info_base_parser)
    return info_base_parser


@register_command(make_info_base)
def info(args):
    logger = get_logger()
    network_info = get_network_info(args.model_name)
    model_name = network_info.network.network_name
    logger.info(f"Start run for network {model_name} ...")
    info_model(model_name, network_info, logger)
