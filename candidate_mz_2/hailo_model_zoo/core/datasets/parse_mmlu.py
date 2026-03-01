import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY


@DATASET_FACTORY.register(name="mmlu")
def parse_mmlu(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information"""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "prompt": tf.io.FixedLenFeature([], tf.string),
            "answer_gt": tf.io.FixedLenFeature([], tf.string),
        },
    )

    prompt = tf.cast(features["prompt"], tf.string)
    answer_gt = tf.cast(features["answer_gt"], tf.string)

    image_info = {"prompt": prompt, "answer_gt": answer_gt}
    return [prompt, image_info]
