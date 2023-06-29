try:
    from hailo_platform import PcieDevice, HailoRTException
    PLATFORM_AVAILABLE = True
except ModuleNotFoundError:
    PLATFORM_AVAILABLE = False


PROFILER_MODE_NAMES = {'pre_placement', 'post_placement'}

TARGETS = [
    'hailo8',
    'full_precision',
    'emulator',
]


DEVICE_NAMES = set()
if PLATFORM_AVAILABLE:
    try:
        devices = PcieDevice.scan_devices()
        DEVICE_NAMES.update([str(name) for name in devices])
    except HailoRTException:
        # Ignore HailoRT exception when the driver is not installed
        pass
