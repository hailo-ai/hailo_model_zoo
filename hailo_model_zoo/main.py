#!/usr/bin/env python
import argparse
import importlib

# we try to minimize imports to make 'main.py --help' responsive. So we only import definitions.
import hailo_model_zoo.plugin
from hailo_model_zoo.base_parsers import (
    make_evaluation_base,
    make_hef_base,
    make_optimization_base,
    make_parsing_base,
    make_profiling_base,
)
from hailo_model_zoo.utils.cli_utils import HMZ_COMMANDS
from hailo_model_zoo.utils.plugin_utils import iter_namespace
from hailo_model_zoo.utils.version import get_version

discovered_plugins = {
    name: importlib.import_module(name) for finder, name, ispkg in iter_namespace(hailo_model_zoo.plugin)
}


def _create_args_parser():
    # --- create shared arguments parsers
    parsing_base_parser = make_parsing_base()
    optimization_base_parser = make_optimization_base()
    hef_base_parser = make_hef_base()
    profile_base_parser = make_profiling_base()
    evaluation_base_parser = make_evaluation_base()
    version = get_version("hailo_model_zoo")

    # --- create per action subparser
    parser = argparse.ArgumentParser(epilog="Example: hailomz parse resnet_v1_50")
    parser.add_argument("--version", action="version", version=f"Hailo Model Zoo v{version}")
    # can't set the entry point for each subparser as it forces us to add imports which slow down the startup time.
    # instead we'll check the 'command' argument after parsing
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "parse",
        parents=[parsing_base_parser],
        help="model translation of the input model into Hailo's internal representation.",
    )

    subparsers.add_parser(
        "optimize",
        parents=[parsing_base_parser, optimization_base_parser],
        help="run model optimization which includes numeric translation of \
                                the input model into a compressed integer representation.",
    )

    compile_help = (
        "run the Hailo compiler to generate the Hailo Executable Format file (HEF)"
        " which can be executed on the Hailo hardware."
    )
    subparsers.add_parser(
        "compile",
        parents=[parsing_base_parser, optimization_base_parser],
        help=compile_help,
    )

    profile_help = (
        "generate profiler report of the model."
        " The report contains information about your model and expected performance on the Hailo hardware."
    )
    subparsers.add_parser(
        "profile",
        parents=[
            parsing_base_parser,
            optimization_base_parser,
            hef_base_parser,
            profile_base_parser,
        ],
        help=profile_help,
    )

    subparsers.add_parser(
        "eval",
        parents=[
            parsing_base_parser,
            optimization_base_parser,
            hef_base_parser,
            evaluation_base_parser,
        ],
        help="infer the model using the Hailo Emulator or the Hailo hardware and produce the model accuracy.",
    )

    # add parsers for plugins
    for command in HMZ_COMMANDS:
        command_parser = command.parser_fn()
        subparsers.add_parser(command.name, parents=[command_parser], help=command_parser.description)
    return parser


def run(args):
    # search for commands from plugins
    command_to_handler = {command.name: command.fn for command in HMZ_COMMANDS}
    if args.command in command_to_handler:
        return command_to_handler[args.command](args)

    # we make sure to only import these now to keep loading & plugins fast
    from hailo_model_zoo.main_driver import compile, evaluate, optimize, parse, profile

    handlers = {
        "parse": parse,
        "optimize": optimize,
        "compile": compile,
        "profile": profile,
        "eval": evaluate,
    }

    return handlers[args.command](args)


def main():
    parser = _create_args_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    # from this point we can import heavy modules
    run(args)


if __name__ == "__main__":
    main()
