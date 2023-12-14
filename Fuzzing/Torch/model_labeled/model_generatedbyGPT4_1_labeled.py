import tensorflow as tf
import numpy as np

class PreprocessAndCalculateModel(tf.keras.Model):
    def __init__(self):
        super(PreprocessAndCalculateModel, self).__init__()
        self.minimum_layer = tf.keras.layers.Minimum()
        self.maximum_layer = tf.keras.layers.Maximum()
        self.dense = tf.keras.layers.Dense(10, activation=tf.keras.activations.exponential, kernel_initializer=tf.initializers.he_normal())
        self.used_apis = [
            'tf.reduce_sum',
            'tf.concat',
            'tf.keras.layers.Minimum()',
            'tf.keras.layers.Maximum()',
            'tf.keras.layers.Dense',
            'tf.experimental.numpy.exp',
            'tf.experimental.numpy.arccos',
            'tf.experimental.numpy.nextafter',
            'tf.linalg.eye',
            'tf.GradientTape',
            'tf.divide'
        ]
    def call(self, inputs):
        # Record input-output of each API
        api_outputs = {}

        api_outputs['pre_tf.reduce_sum(inputs, axis=-1)'] = inputs
        x = tf.reduce_sum(inputs, axis=-1)
        api_outputs['tf.reduce_sum(inputs, axis=-1)'] = x

        api_outputs['pre_tf.concat([x, x], axis=-1)'] = x
        x = tf.concat([x, x], axis=-1)
        api_outputs['tf.concat([x, x], axis=-1)'] = x

        api_outputs['pre_self.minimum_layer([x, x])'] = x
        x = self.minimum_layer([x, x])
        api_outputs['self.minimum_layer([x, x])'] = x

        api_outputs['pre_self.maximum_layer([x, x])'] = x
        x = self.maximum_layer([x, x])
        api_outputs['self.maximum_layer([x, x])'] = x

        api_outputs['pre_self.dense(x)'] = x
        x = self.dense(x)
        api_outputs['self.dense(x)'] = x

        api_outputs['pre_tf.experimental.numpy.exp(x)'] = x
        x = tf.experimental.numpy.exp(x)
        api_outputs['tf.experimental.numpy.exp(x)'] = x

        api_outputs['pre_tf.experimental.numpy.arccos(x)'] = x
        x = tf.experimental.numpy.arccos(x)
        api_outputs['tf.experimental.numpy.arccos(x)'] = x

        api_outputs['pre_tf.experimental.numpy.nextafter(x, x + 1)'] = x
        x = tf.experimental.numpy.nextafter(x, x + 1)
        api_outputs['tf.experimental.numpy.nextafter(x, x + 1)'] = x

        api_outputs['pre_tf.linalg.eye(tf.shape(x)[0])'] = x
        x = tf.linalg.eye(tf.shape(x)[0])
        api_outputs['tf.linalg.eye(tf.shape(x)[0])'] = x

        api_outputs['pre_tf.divide(x, x)'] = x
        api_outputs['pre_tape.jacobian(z, x)'] = x
        with tf.GradientTape() as tape:
            tape.watch(x)
            z = tf.divide(x, x)
        jaco = tape.jacobian(z, x)
        api_outputs['tf.divide(x, x)'] = z
        api_outputs['tape.jacobian(z, x)'] = jaco

        # List of used APIs
        used_apis_sorted = sorted(set(self.used_apis), key=self.used_apis.index)

        return jaco, used_apis_sorted