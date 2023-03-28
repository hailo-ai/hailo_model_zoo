from hailo_sdk_common.targets.inference_targets import SdkPartialNumeric, SdkFPOptimized
from hailo_sdk_client import InferenceContext

try:
    from hailo_platform import PcieDevice, HailoRTException
    PLATFORM_AVAILABLE = True
except ModuleNotFoundError:
    PLATFORM_AVAILABLE = False

TARGETS = {
    'hailo8': PcieDevice if PLATFORM_AVAILABLE else None,
    'full_precision': SdkFPOptimized,
    'emulator': SdkPartialNumeric,
}

INFERENCE_TARGETS = {
    'hailo8': InferenceContext.SDK_HAILO_HW,
    'full_precision': InferenceContext.SDK_FP_OPTIMIZED,
    'emulator': InferenceContext.SDK_QUANTIZED,
}

DEVICES = {}
DEVICE_NAMES = set()
if PLATFORM_AVAILABLE:
    try:
        devices = PcieDevice.scan_devices()
        TARGETS.update({str(name): lambda: PcieDevice(name) for name in devices})
        INFERENCE_TARGETS.update({str(name): InferenceContext.SDK_HAILO_HW for name in devices})
        DEVICES.update({str(name): name for name in devices})
        DEVICE_NAMES.update([str(name) for name in devices])
    except HailoRTException:
        # Ignore HailoRT exception when the driver is not installed
        pass
