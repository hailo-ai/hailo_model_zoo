#!/usr/bin/env python

from setuptools import find_packages, setup

try:
    import hailo_sdk_client
except ModuleNotFoundError:
    raise ModuleNotFoundError("hailo_sdk_client was not installed or you are not "
                              "in the right virtual env.")
try:
    import hailo_platform
except ModuleNotFoundError:
    raise ModuleNotFoundError("hailo_platform was not installed or you are not "
                              "in the right virtual env.")


def main():

    reqs = ['Cython',
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
            'scikit-image==0.17.2',
            'wget==3.2',
            'tf-object-detection==0.0.3']

    model_zoo_version = "1.0.0"

    setup(
        name='hailo_model_zoo',
        version=model_zoo_version,
        description='Hailo machine learning utilities and examples',
        url='https://hailo.ai/',
        author='Hailo team',
        author_email='contact@hailo.ai',
        entry_points={},
        license='',
        packages=find_packages(),
        install_requires=reqs,
        zip_safe=False
    )


if __name__ == '__main__':
    main()
