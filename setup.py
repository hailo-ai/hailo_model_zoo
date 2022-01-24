#!/usr/bin/env python

from setuptools import find_packages, setup

try:
    import hailo_sdk_client  # noqa F401
except ModuleNotFoundError:
    raise ModuleNotFoundError("hailo_sdk_client was not installed or you are not "
                              "in the right virtualenv.\n"
                              "In case you are not an Hailo customer please visit us at https://hailo.ai/")
try:
    import hailo_platform  # noqa F401
except ModuleNotFoundError:
    raise ModuleNotFoundError("hailo_platform was not installed or you are not "
                              "in the right virtualenv.\n"
                              "In case you are not an Hailo customer please visit us at https://hailo.ai/")


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
            'motmetrics==1.2.0',
            'omegaconf==2.1.0',
            'pillow==8.1.2',
            'detection-tools==0.3',
            'scikit-image==0.17.2']

    model_zoo_version = "1.4.0"

    package_data = {
        "hailo_model_zoo": [
            "cfg/base/*.yaml", "cfg/networks/*.yaml", "cfg/alls/*.alls", "datasets/*",
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
        entry_points={},
        license='MIT',
        packages=find_packages(),
        install_requires=reqs,
        zip_safe=False,
        package_data=package_data
    )


if __name__ == '__main__':
    main()
