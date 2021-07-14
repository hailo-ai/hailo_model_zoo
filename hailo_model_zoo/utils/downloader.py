"""
model files downloader

Usage:
    >>> from downloader import download
    >>> from logging import getLogger
    >>> model_files_dir = '.'
    >>> hailo_storage = 'https://hailo-modelzoo-pub.s3.eu-west-2.amazonaws.com/'
    >>> model_path = 'Classification/mobilenet_v1/pretrained/mobilenet_v1_1_0_224.ckpt.zip'
    >>> download(hailo_storage+model_path, model_files_dir, getLogger())
"""
import logging
import zipfile

from pathlib import Path
from requests import get
from tqdm.auto import tqdm
from typing import Union


def _download(url: str, dst: Path) -> None:
    resp = get(url, allow_redirects=True, stream=True)

    with dst.open('wb') as fout:
        with tqdm(
            desc=dst.name,
            miniters=1,
            total=int(resp.headers.get('content-length', 0)),
            unit='B',
            unit_divisor=1024,
            unit_scale=True,
        ) as progress_bar:
            for chunk in resp.iter_content(chunk_size=4096):
                fout.write(chunk)
                progress_bar.update(len(chunk))


def download(url: str, dst_dir: Union[str, Path], logger: logging.Logger) -> str:
    """downloads a file from given url, and returns the downloaded file name"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = Path(dst_dir) / Path(url).name

    if not(dst.exists() and dst.is_file()):
        logger.debug(f'downloading {url} into {dst_dir}')
        _download(url, dst)
    else:
        logger.debug(f'{dst.name} already exists inside {dst_dir}. Skipping download')

    if len(list(Path('/'.join(dst.parts[:-1])).iterdir())) == 1:
        logger.debug(f'unzipping {dst} into {dst_dir}')
        with zipfile.ZipFile(dst, 'r') as zip_fp:
            zip_fp.extractall(dst_dir)

    return dst.name
