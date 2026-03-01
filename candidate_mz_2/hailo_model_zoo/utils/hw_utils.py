from hailo_sdk_client import InferenceContext
from hailo_sdk_common.targets.inference_targets import SdkFPOptimized, SdkPartialNumeric

from hailo_model_zoo.utils.platform_discovery import PLATFORM_AVAILABLE

if PLATFORM_AVAILABLE:
    from hailo_platform import Device, HailoRTException
    from hailo_platform.pyhailort._pyhailort import HailoRTStatusException

TARGETS = {
    "hardware": Device if PLATFORM_AVAILABLE else None,
    "full_precision": SdkFPOptimized,
    "emulator": SdkPartialNumeric,
}

INFERENCE_TARGETS = {
    "hardware": InferenceContext.SDK_HAILO_HW,
    "full_precision": InferenceContext.SDK_FP_OPTIMIZED,
    "emulator": InferenceContext.SDK_QUANTIZED,
}

DEVICES = {}
DEVICE_NAMES = set()
if PLATFORM_AVAILABLE:
    try:
        devices = Device.scan()
        TARGETS.update({str(name): lambda: Device(name) for name in devices})  # noqa: B023
        INFERENCE_TARGETS.update({str(name): InferenceContext.SDK_HAILO_HW for name in devices})
        DEVICES.update({str(name): name for name in devices})
        DEVICE_NAMES.update([str(name) for name in devices])
    except (HailoRTException, HailoRTStatusException):
        # Ignore HailoRT exception when the driver is not installed
        pass
