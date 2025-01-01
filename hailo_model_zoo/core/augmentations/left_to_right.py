import tensorflow as tf


class AugmentedLRModel(tf.keras.Model):
    def __init__(self, m):
        super().__init__()
        self._m = m

    def call(self, x):
        flipped = tf.image.flip_left_right(x)
        return tf.concat([self._m(x), self._m(flipped)], axis=-1)
