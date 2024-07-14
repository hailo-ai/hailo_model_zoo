import contextlib

import cv2


@contextlib.contextmanager
def VideoCapture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


@contextlib.contextmanager
def VideoWriter(*args, **kwargs):
    cap = cv2.VideoWriter(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()
