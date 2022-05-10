from hailo_sdk_common.targets.inference_targets import SdkNative, SdkPartialNumeric

try:
    from hailo_platform.drivers.hailort.pyhailort import HailoRTException
    from hailo_platform.drivers.hw_object import PcieDevice
    PLATFORM_AVAILABLE = True
except ModuleNotFoundError:
    PLATFORM_AVAILABLE = False

TARGETS = {'hailo8': PcieDevice if PLATFORM_AVAILABLE else None,
           'full_precision': SdkNative,
           'emulator': SdkPartialNumeric,
           }
DEVICE_NAMES = set()
if PLATFORM_AVAILABLE:
    try:
        devices = PcieDevice.scan_devices()
        TARGETS.update({str(name): lambda: PcieDevice(name) for name in devices})
        DEVICE_NAMES.update([str(name) for name in devices])
    except HailoRTException:
        # Ignore HailoRT exception when the driver is not installed
        pass
