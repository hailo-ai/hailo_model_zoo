try:
    # check if platform is available
    import hailo_platform  # noqa: F401

    with open("/proc/modules", "r") as f:
        if f.read().find("hailo") < 0:
            raise ModuleNotFoundError("Hailo driver is not installed")
    PLATFORM_AVAILABLE = True
except ModuleNotFoundError:
    PLATFORM_AVAILABLE = False
