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
        api_outputs = {}

        api_outputs['pre_self.input_layer(inputs)'] = inputs
        x = self.input_layer(inputs)
        api_outputs['self.input_layer(inputs)'] = x

        api_outputs['pre_tape.jacobian(y, x)'] = x
        api_outputs['pre_tf.py_function(lambda x: tf.math.log(x), [x], x.dtype)'] = x
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = tf.py_function(lambda x: tf.math.log(x), [x], x.dtype)
        jaco = tape.jacobian(y, x)
        api_outputs['tf.py_function(lambda x: tf.math.log(x), [x], x.dtype)'] = y
        api_outputs['tape.jacobian(y, x)'] = jaco

        api_outputs['pre_tf.reshape(jaco, [4, -1, 3])'] = jaco
        x = tf.reshape(jaco, [4, -1, 3])  # Reshape for separable_conv1d
        api_outputs['tf.reshape(jaco, [4, -1, 3])'] = x

        api_outputs['preself.separable_conv1d(x)'] = x
        x = self.separable_conv1d(x)
        api_outputs['self.separable_conv1d(x)'] = x

        api_outputs['pre_self.max_pooling(x)'] = x
        x = self.max_pooling(x)
        api_outputs['self.max_pooling(x)'] = x

        api_outputs['pre_self.flatten(x)'] = x
        x = self.flatten(x)
        api_outputs['self.flatten(x)'] = x

        api_outputs['pre_self.dense(x)'] = x
        x = self.dense(x)
        api_outputs['self.dense(x)'] = x

        # Additional API usages from the list
        api_outputs['pre_tf.maximum(x, 0.5)'] = x
        x = tf.maximum(x, 0.5)
        api_outputs['tf.maximum(x, 0.5)'] = x

        api_outputs['pre_tf.roll(x, shift=1, axis=1)'] = x
        x = tf.roll(x, shift=1, axis=1)
        api_outputs['tf.roll(x, shift=1, axis=1)'] = x

        api_outputs['pre_tf.experimental.numpy.logical_or(x > 0, x < 0)'] = x
        x = tf.experimental.numpy.logical_or(x > 0, x < 0)
        api_outputs['tf.experimental.numpy.logical_or(x > 0, x < 0)']    = x

        api_outputs['pre_tf.experimental.numpy.nanmean(x)'] = x
        x = tf.experimental.numpy.nanmean(x)
        api_outputs['tf.experimental.numpy.nanmean(x)'] = x

        api_outputs['pre_tf.experimental.numpy.tan(x)'] = x
        x = tf.experimental.numpy.tan(x)
        api_outputs['tf.experimental.numpy.tan(x))'] = x

        api_outputs['pre_tf.math.round(x)'] = x
        x = tf.math.round(x)
        api_outputs['tf.math.round(x)'] = x

        # Final output calculation
        output = x  # Final operation

        # List of used APIs
        used_apis_sorted = sorted(set(self.used_apis), key=self.used_apis.index)

        return output, used_apis_sorted
