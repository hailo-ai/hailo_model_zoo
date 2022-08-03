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
import shutil
import tempfile
import zipfile

from pathlib import Path
from requests import get, Session, Response
from tqdm.auto import tqdm
from typing import Union, Optional, Generator, IO
from urllib.parse import urlparse
_DEFAULT_CHUNK_SIZE = 4096


def _write_with_progress(fout: IO, iterator: Generator, *,
                         desc: str, content_length: Optional[int] = 0):
    with tqdm(
        desc=desc,
        miniters=1,
        total=content_length,
        unit='B',
        unit_divisor=1024,
        unit_scale=True,
    ) as progress_bar:
        for chunk in iterator:
            fout.write(chunk)
            progress_bar.update(len(chunk))


def download_to_file(url: str, fout: IO, desc: Optional[str] = None) -> None:
    desc = desc or Path(urlparse(url).path).name
    resp = get(url, allow_redirects=True, stream=True)
    resp.raise_for_status()

    _write_with_progress(fout, resp.iter_content(chunk_size=_DEFAULT_CHUNK_SIZE),
                         desc=desc, content_length=int(resp.headers.get('content-length', 0)))


def download_file(url: str, dst: Path = Path('.'), desc: Optional[str] = None) -> None:
    if not dst.is_dir():
        # if destination is a file, we use its name
        filename = dst.name
    else:
        # if destination is a directory we try to infer the file name.
        netpath = urlparse(url).path
        filename = Path(netpath).name
        dst = dst / filename

    desc = desc or filename

    with tempfile.TemporaryDirectory() as temp_dir:
        with open(f'{temp_dir}/temp_file', 'wb') as outfile:
            download_to_file(url, outfile, desc=desc)
        shutil.move(outfile.name, str(dst))
    return dst


def download(url: str, dst_dir: Union[str, Path], logger: logging.Logger) -> str:
    """downloads a file from given url, and returns the downloaded file name"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = Path(dst_dir) / Path(url).name

    if not(dst.exists() and dst.is_file()):
        logger.debug(f'downloading {url} into {dst_dir}')
        download_file(url, dst)
    else:
        logger.debug(f'{dst.name} already exists inside {dst_dir}. Skipping download')

    if len(list(Path('/'.join(dst.parts[:-1])).iterdir())) == 1:
        logger.debug(f'unzipping {dst} into {dst_dir}')
        with zipfile.ZipFile(dst, 'r') as zip_fp:
            zip_fp.extractall(dst_dir)

    return dst.name


def _get_auth_token(response: Response):
    for k, v in response.cookies.items():
        if k.startswith('download_warning'):
            return v
    return 't'


def download_from_drive(url: str, fp: IO, desc: Optional[str] = None):
    with Session() as session:
        with session.get(url, stream=True) as response:
            response.raise_for_status()
            auth_token = _get_auth_token(response)
        download_url = f'{url}&confirm={auth_token}'
        with session.get(download_url, stream=True) as response:
            response.raise_for_status()
            _write_with_progress(fp, response.iter_content(_DEFAULT_CHUNK_SIZE),
                                 content_length=int(response.headers.get('content-length', 0)),
                                 desc=desc)
