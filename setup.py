#!/usr/bin/env python

from setuptools import find_packages, setup


import subprocess
check_dfc_installed = subprocess.run(
    "pip show hailo-dataflow-compiler".split(),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
if check_dfc_installed.stderr:
    raise ModuleNotFoundError("hailo_sdk_client was not installed or you are not "
                              "in the right virtualenv.\n"
                              "In case you are not an Hailo customer please visit us at https://hailo.ai/")


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


def main():

    reqs = ['Cython',
            'imageio==2.9.0',
            'matplotlib',
            'numpy',
            'opencv-python',
            'scipy',
            'scikit-learn',
            'termcolor',
            'tqdm',
            'pycocotools',
            'lap==0.4.0',
            'motmetrics==1.2.5',
            'omegaconf==2.3.0',
            'pillow<=9.2.0',
            'detection-tools==0.3',
            'scikit-image==0.19.3',
            'torch==1.11.0',
            'torchmetrics==1.2.0']

    model_zoo_version = "2.10.0"

    package_data = {
        "hailo_model_zoo": [
            "cfg/base/*.yaml", "cfg/networks/*.yaml",
            "cfg/alls/*/*.alls", "datasets/*",
            "cfg/multi-networks/*.yaml", "cfg/multi-networks/*.yaml",
            "core/postprocessing/*.json",
            "core/postprocessing/src/*.cc",
            "core/postprocessing/cython_utils/cython_nms.pyx",
            "core/eval/widerface_evaluation_external/box_overlaps.pyx",
        ]
    }

    setup(
        name='hailo_model_zoo',
        version=model_zoo_version,
        description='Hailo machine learning utilities and examples',
        url='https://hailo.ai/',
        author='Hailo team',
        author_email='hailo_model_zoo@hailo.ai',
        entry_points={"console_scripts": ["hailomz=hailo_model_zoo.main:main"]},
        license='MIT',
        packages=find_packages(),
        install_requires=reqs,
        zip_safe=False,
        package_data=package_data
    )


if __name__ == '__main__':
    main()
