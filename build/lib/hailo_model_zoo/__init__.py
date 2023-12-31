try:
    import cpuinfo
    cpu_flags = cpuinfo.get_cpu_info()['flags']
except Exception:
    cpu_flags = None
    print("Warning! Unable to query CPU for the list of supported features.")

if cpu_flags is not None and 'avx' not in cpu_flags:
    print("""
        This CPU does not support `avx` instructions, and they are needed to run Tensorflow.
        It is recommended to run the Dataflow Compiler on another host.
        Another option is to compile Tensorflow from sources without `avx` instructions.
    """)


from hailo_model_zoo.utils.version import get_version

__version__ = get_version('hailo_model_zoo')
