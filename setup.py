#!/usr/bin/env python
from importlib.metadata import PackageNotFoundError, version

from setuptools import find_packages, setup

CUR_DFC_VERSION = "v3.32.0"
package_name = "hailo-dataflow-compiler"

try:
    dfc_version = version(package_name)
    if dfc_version != CUR_DFC_VERSION:
        print(
            f"Warning! The current version of the Dataflow Compiler is {dfc_version}.\n"
            f"Current Hailo-Model-Zoo works best with DFC version {CUR_DFC_VERSION}. Please consider updating your DFC"
        )
except PackageNotFoundError:
    raise PackageNotFoundError(
        f"\nThe Dataflow Compiler package {package_name!r} was not found.\n"
        f"Please verify working in the correct virtualenv.\n"
        f"If you are not an Hailo customer, please visit us at https://hailo.ai/"
    ) from None

try:
    import cpuinfo

    cpu_flags = cpuinfo.get_cpu_info()["flags"]
except Exception:
    cpu_flags = None
    print("Warning! Unable to query CPU for the list of supported features.")

if cpu_flags is not None and "avx" not in cpu_flags:
    print("""
        This CPU does not support `avx` instructions, and they are needed to run Tensorflow.
        It is recommended to run the Dataflow Compiler on another host.
        Another option is to compile Tensorflow from sources without `avx` instructions.
    """)


def main():
    reqs = [
        "numba==0.59.0",
        "imageio==2.22.4",
        "matplotlib",
        "numpy",
        "opencv-python",
        "scipy",
        "scikit-learn",
        "termcolor",
        "tqdm",
        "pycocotools",
        "lap==0.5.12",
        "motmetrics==1.2.5",
        "omegaconf==2.3.0",
        "pillow<=9.3.0",
        "detection-tools==0.3",
        "scikit-image==0.20.0",
        "nuscenes-devkit",
        "pyquaternion==0.9.9",
        "Shapely>=2.0.0",
    ]

    model_zoo_version = "2.16.0"

    package_data = {
        "hailo_model_zoo": [
            "cfg/base/*.yaml",
            "cfg/networks/*.yaml",
            "cfg/alls/**/*.alls",
            "datasets/*",
            "cfg/cascades/**",
            "cfg/multi-networks/*.yaml",
            "cfg/multi-networks/*.yaml",
            "core/postprocessing/*.json",
            "core/postprocessing/src/*.cc",
            "core/postprocessing/cython_utils/cython_nms.pyx",
            "core/eval/widerface_evaluation_external/box_overlaps.pyx",
        ]
    }

    setup(
        name="hailo_model_zoo",
        version=model_zoo_version,
        description="Hailo machine learning utilities and examples",
        url="https://hailo.ai/",
        author="Hailo team",
        author_email="hailo_model_zoo@hailo.ai",
        entry_points={"console_scripts": ["hailomz=hailo_model_zoo.main:main"]},
        license="MIT",
        packages=find_packages(),
        install_requires=reqs,
        zip_safe=False,
        package_data=package_data,
    )


if __name__ == "__main__":
    main()
