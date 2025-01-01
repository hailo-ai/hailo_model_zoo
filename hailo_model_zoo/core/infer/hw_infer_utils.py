from typing import List

import tensorflow as tf
from contextlib2 import ExitStack

from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)

from hailo_sdk_client import ClientRunner


class HefWrapper:
    def __init__(self, path, device=None, input_format_type=None, batch_size=None):
        with ExitStack() as stack:
            target = device or stack.enter_context(VDevice())
            hef = HEF(path)

            configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)

            if batch_size is not None:
                for params in configure_params.values():
                    params.batch_size = batch_size

            network_groups = target.configure(hef, configure_params)
            network_group = network_groups[0]
            network_group_params = network_group.create_params()
            input_vstreams_params = InputVStreamParams.make(
                network_group, format_type=input_format_type or FormatType.FLOAT32
            )
            output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

            infer_cascade = stack.enter_context(
                InferVStreams(network_group, input_vstreams_params, output_vstreams_params)
            )
            self.activate = lambda: network_group.activate(network_group_params)

            self.infer_cascade = infer_cascade
            self.stack = stack.pop_all()

    def __call__(self, inp):
        with self.activate():
            return self.infer_cascade.infer(inp)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stack.close()


class HefModel:
    def __init__(self, hef_wrapper: HefWrapper, inputs: List[str], outputs: List[str]):
        self.model = hef_wrapper
        self.inputs = inputs
        self.outputs = outputs

    @staticmethod
    def from_runner(hef_wrapper: HefWrapper, runner: ClientRunner):
        hn_model = runner.get_hn_model()
        return HefModel(
            hef_wrapper,
            [layer.name for layer in hn_model.get_input_layers()],
            [layer.name for layer in hn_model.get_real_output_layers()],
        )

    def call(self, *inp):
        inp = dict(zip(self.inputs, inp))
        x = self.model(inp)
        x = [x[output] for output in self.outputs]
        return x

    def __call__(self, args):
        if isinstance(args, dict):
            args = [args[input_name] for input_name in self.inputs]
        else:
            args = [args]
        return tf.numpy_function(self.call, args, tf.float32)
