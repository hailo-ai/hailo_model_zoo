import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY

_SIZE = 224


@DATASET_FACTORY.register(name="lagenda")
def parse_lagenda_record(serialized_example):
    """Lagenda age/gender record. Stores the pre-letterboxed 224x224 face + person
    crops (built by create_lagenda_tfrecord.py via MiVOLO's reader) so preprocessing
    just concatenates + masks. GT fields (age, is_female_int) match AgeGenderEval."""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "age": tf.io.FixedLenFeature([], tf.int64),
            "is_female_int": tf.io.FixedLenFeature([], tf.int64),
            "image_id": tf.io.FixedLenFeature([], tf.int64),
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "face_jpeg": tf.io.FixedLenFeature([], tf.string),
            "person_jpeg": tf.io.FixedLenFeature([], tf.string),
        },
    )
    face = tf.reshape(tf.cast(tf.image.decode_jpeg(features["face_jpeg"], channels=3), tf.uint8), [_SIZE, _SIZE, 3])
    person = tf.reshape(tf.cast(tf.image.decode_jpeg(features["person_jpeg"], channels=3), tf.uint8), [_SIZE, _SIZE, 3])
    image_info = {
        "image_name": tf.cast(features["image_name"], tf.string),
        "height": _SIZE,
        "width": _SIZE,
        "image_id": tf.cast(features["image_id"], tf.int32),
        "age": tf.cast(features["age"], tf.int32),
        "is_female_int": tf.cast(features["is_female_int"], tf.int32),
        "person": person,
    }
    return [face, image_info]
