from hailo_model_zoo.utils.platform_discovery import PLATFORM_AVAILABLE

if PLATFORM_AVAILABLE:
    from hailo_model_zoo.core.infer.hw_infer_utils import HefModel, HefWrapper

from hailo_sdk_client import ClientRunner, InferenceContext

from hailo_model_zoo.core.main_utils import (
    get_network_info,
    get_postprocessing_callback,
    make_eval_callback,
    make_evalset_callback,
    make_preprocessing,
)
from hailo_model_zoo.utils.hw_utils import DEVICES, INFERENCE_TARGETS


class Model:
    def __init__(self, network_name, target, har, hef):
        self.info = get_network_info(network_name)
        self.target = INFERENCE_TARGETS[target]
        self.device_info = DEVICES.get(target)
        self.har = har
        self.hef = hef
        self.runner = ClientRunner(har=self.har)

    def get_keras_model(self, stack, device=None, hef_kwargs=None):
        hef_kwargs = hef_kwargs or {}
        if self.target is InferenceContext.SDK_HAILO_HW:
            if not PLATFORM_AVAILABLE:
                raise ValueError(
                    (
                        f"Chosen target for {self.info.network.network_name} is {self.target} "
                        "but hailo_platform is not available",
                    )
                )
            return HefModel.from_runner(
                stack.enter_context(HefWrapper(self.hef, device=device, **hef_kwargs)), self.runner
            )
        context = stack.enter_context(self.runner.infer_context(self.target, self.device_info))
        return self.runner.get_keras_model(context)

    def get_preproc(self):
        return make_preprocessing(self.runner, self.info)

    def get_postproc(self):
        return get_postprocessing_callback(self.runner, self.info)

    def get_dataset(self):
        preproc = self.get_preproc()
        return make_evalset_callback(self.info, preproc, override_path=None)

    def get_eval(self):
        return make_eval_callback(self.info, self.runner, show_results_per_class=False, logger=None)

    def get_input_shapes(self, *, ignore_conversion=True):
        return self.runner.get_hn_model().get_input_shapes(ignore_conversion=ignore_conversion)
