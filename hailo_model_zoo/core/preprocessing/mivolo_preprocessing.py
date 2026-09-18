import tensorflow as tf

from hailo_model_zoo.core.factory import PREPROCESS_FACTORY


@PREPROCESS_FACTORY.register(name="mivolo")
def mivolo_preprocessing(image, image_info=None, output_height=224, output_width=224, mivolo_mode="facebody", **kwargs):
    """Build MiVOLO's 6-channel input = concat(face RGB, person RGB).

    The face + person crops are pre-letterboxed to output_height x output_width by the
    TFRecord creator (using MiVOLO's own reader, so crops match the reference exactly).
    `image` is the face crop; `image_info["person"]` is the person crop.

    3 deployment modes (a missing crop is fed as black, exactly as MiVOLO trains):
      facebody -> both crops; body -> face=black; face -> person=black.
    On the folder-input path (ImageFeed, e.g. ``hailomz infer/eval --data-path <dir>``)
    there is no person crop, so it falls back to black - i.e. the face-only mode the
    model already supports - rather than raising a KeyError.
    Normalization is folded into the net (parser.normalization_params), so this returns
    a raw [0,255] float tensor.
    """
    channels = int(kwargs.get("channels", 6))
    face = tf.cast(image, tf.float32)
    if image_info is not None and "person" in image_info:
        person = tf.cast(image_info["person"], tf.float32)
    else:
        person = tf.zeros_like(face)

    if mivolo_mode == "body":
        face = tf.zeros_like(face)
    elif mivolo_mode == "face":
        person = tf.zeros_like(person)

    # Crops are already output_height x output_width from the TFRecord creator; resize
    # defensively so an off-size input fails here diagnosably rather than in set_shape.
    face = tf.image.resize(face, (output_height, output_width))
    person = tf.image.resize(person, (output_height, output_width))

    out = tf.concat([face, person], axis=-1)  # [output_height, output_width, channels]
    out.set_shape([output_height, output_width, channels])
    if image_info is not None:
        image_info["img_orig"] = tf.cast(image, tf.uint8)
    return out, image_info
