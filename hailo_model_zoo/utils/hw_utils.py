from hailo_sdk_common.targets.inference_targets import SdkNative, SdkPartialNumeric

try:
    from hailo_platform.drivers.hw_object import PcieDevice
    TARGETS = {
        'hailo8': PcieDevice
    }
except ModuleNotFoundError:
    TARGETS = {'hailo8': None}

TARGETS.update({
    'full_precision': SdkNative,
    'emulator': SdkPartialNumeric})
