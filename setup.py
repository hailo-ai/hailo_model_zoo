#!/usr/bin/env python

from glob import glob
from setuptools import find_packages, setup

try:
    import hailo_sdk_client  # noqa F401
except ModuleNotFoundError:
    raise ModuleNotFoundError("hailo_sdk_client was not installed or you are not "
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

    model_zoo_version = "2.1.0"

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

    # Get a list of all directories that contain a markdown file:
    md_locs = list(set(['/'.join(path.split('/')[:-1]) for path in glob('./**/*.md', recursive=True)]))
    # Create a data_files structure for all markdowns:
    # This is a list of tuples of the form (output_dir, [path/to/file_1, path/to/file_2, ...])
    doc_dir = 'hailo_model_zoo_doc'
    data_files = [(f'{doc_dir}/{md_loc}', glob(f'{md_loc}/*.md')) for md_loc in md_locs if "venv" not in md_loc]

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
        package_data=package_data,
        data_files=data_files
    )


if __name__ == '__main__':
    main()
