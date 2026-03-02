#!/usr/bin/env python3
"""
Parser for COCO TFRecord dataset with text embeddings.

This module provides parsing functions for TFRecord files created by
create_coco_tfrecord_with_text_embeddings.py. It can be used as a
reference for integrating with the Hailo model zoo framework.
"""

import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY


@DATASET_FACTORY.register(name="coco_with_text_embeddings")
def parse_coco_with_text_embeddings(serialized_example):
    """
    Parse serialized example of TfRecord with text embeddings.

    Args:
        serialized_example: Serialized tf.train.Example

    Returns:
        Tuple of (image, image_info) where image_info contains text_embeddings
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
            "image_id": tf.io.FixedLenFeature([], tf.int64),
            "xmin": tf.io.VarLenFeature(tf.float32),
            "xmax": tf.io.VarLenFeature(tf.float32),
            "ymin": tf.io.VarLenFeature(tf.float32),
            "ymax": tf.io.VarLenFeature(tf.float32),
            "area": tf.io.VarLenFeature(tf.float32),
            "category_id": tf.io.VarLenFeature(tf.int64),
            "is_crowd": tf.io.VarLenFeature(tf.int64),
            "num_boxes": tf.io.FixedLenFeature([], tf.int64),
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "image_jpeg": tf.io.FixedLenFeature([], tf.string),
            # Text embeddings fields
            "text_embeddings": tf.io.FixedLenFeature([], tf.string),
            "text_embeddings_shape": tf.io.VarLenFeature(tf.int64),
        },
    )

    # Parse standard COCO fields
    height = tf.cast(features["height"], tf.int32)
    width = tf.cast(features["width"], tf.int32)
    image_id = tf.cast(features["image_id"], tf.int32)
    image_name = tf.cast(features["image_name"], tf.string)
    image = tf.image.decode_jpeg(features["image_jpeg"], channels=3)
    image_shape = tf.stack([height, width, 3])
    image = tf.cast(tf.reshape(image, image_shape), tf.uint8)

    # Parse text embeddings
    # text_embeddings_shape = tf.sparse.to_dense(features["text_embeddings_shape"], default_value=0)
    text_embeddings = tf.io.decode_raw(features["text_embeddings"], tf.float32)
    text_embeddings_shape = tf.stack([1, 80, 512])  # Shape: (1, 1, 80, 512)
    text_embeddings = tf.reshape(text_embeddings, text_embeddings_shape)

    # Build image_info dictionary
    image_info = {
        "image_name": image_name,
        "height": height,
        "width": width,
        "image_id": image_id,
        "text_embeddings": text_embeddings,  # Shape: (1, 1, 80, 512)
        "num_boxes": tf.cast(features["num_boxes"], tf.int32),
        "is_crowd": tf.sparse.to_dense(features["is_crowd"], default_value=0),
        "xmin": tf.sparse.to_dense(features["xmin"], default_value=0),
        "xmax": tf.sparse.to_dense(features["xmax"], default_value=0),
        "ymin": tf.sparse.to_dense(features["ymin"], default_value=0),
        "ymax": tf.sparse.to_dense(features["ymax"], default_value=0),
        "area": tf.sparse.to_dense(features["area"], default_value=0),
        "category_id": tf.sparse.to_dense(features["category_id"], default_value=0),
    }

    return [image, image_info]
