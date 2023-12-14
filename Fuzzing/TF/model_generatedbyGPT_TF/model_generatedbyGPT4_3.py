import tensorflow as tf
import numpy as np

class PreprocessAndCalculateModel(tf.keras.Model):
    def __init__(self):
        super(PreprocessAndCalculateModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(3, 4, 4))
        self.separable_conv1d = tf.keras.layers.SeparableConv1D(32, 3, activation='relu')
        self.max_pooling = tf.keras.layers.MaxPooling1D(2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation=tf.keras.activations.softsign, kernel_initializer=tf.initializers.random_normal())
        self.used_apis = [
            'tf.keras.layers.InputLayer',
            'tf.GradientTape()',
            'tf.py_function',
            'tf.math.log',
            'tf.reshape',
            'tf.keras.layers.SeparableConv1D',
            'tf.keras.layers.MaxPooling1D',
            'tf.keras.layers.Flatten',
            'tf.keras.layers.Dense',
            'tf.maximum',
            'tf.roll',
            'tf.experimental.numpy.logical_or',
            'tf.experimental.numpy.nanmean',
            'tf.experimental.numpy.tan',
            'tf.math.round'
        ]

    def call(self, inputs):
        # Record input-output of each API

        x = self.input_layer(inputs)

        with tf.GradientTape() as tape:
            tape.watch(x)
            y = tf.py_function(lambda x: tf.math.log(x), [x], x.dtype)
        jaco = tape.jacobian(y, x)

        x = tf.reshape(jaco, [4, -1, 3])  # Reshape for separable_conv1d

        x = self.separable_conv1d(x)

        x = self.max_pooling(x)

        x = self.flatten(x)

        x = self.dense(x)

        # Additional API usages from the list
        x = tf.maximum(x, 0.5)

        x = tf.roll(x, shift=1, axis=1)

        x = tf.experimental.numpy.logical_or(x > 0, x < 0)

        x = tf.experimental.numpy.nanmean(x)

        x = tf.experimental.numpy.tan(x)

        x = tf.math.round(x)

        # Final output calculation
        output = x  # Final operation

        # List of used APIs
        used_apis_sorted = sorted(set(self.used_apis), key=self.used_apis.index)

        return output, used_apis_sorted
