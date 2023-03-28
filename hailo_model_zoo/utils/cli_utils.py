from collections import namedtuple
from hailo_model_zoo.utils import path_resolver


def add_model_name_arg(parser, optional=False):
    network_names = list(path_resolver.get_network_names())
    # Setting empty metavar in order to prevent listing the models twice
    nargs = '?' if optional else None
    parser.add_argument('model_name', type=str, nargs=nargs, choices=network_names, metavar='model_name',
                        help='Which network to run. Choices: ' + ', '.join(network_names))


Command = namedtuple('Command', ['name', 'fn', 'parser_fn'])
HMZ_COMMANDS = []


def register_command(parser_factory, *, name=None):
    def _register_inner(func):
        command_name = name or func.__name__
        HMZ_COMMANDS.append(Command(command_name, func, parser_factory))
        return func
    return _register_inner
