from hailo_model_zoo.utils.platform_discovery import PLATFORM_AVAILABLE

TARGETS = [
    "hardware",
    "full_precision",
    "emulator",
]


DEVICE_NAMES = set()
if PLATFORM_AVAILABLE:
    from hailo_platform import Device, HailoRTException
    from hailo_platform.pyhailort._pyhailort import HailoRTStatusException

    try:
        devices = Device.scan()
        DEVICE_NAMES.update([str(name) for name in devices])
    except (HailoRTException, HailoRTStatusException):
        # Ignore HailoRT exception when the driver is not installed
        pass
