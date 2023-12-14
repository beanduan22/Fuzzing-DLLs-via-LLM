import tensorflow as tf
import numpy as np

class PreprocessAndCalculateModel(tf.keras.Model):
    def __init__(self):
        super(PreprocessAndCalculateModel, self).__init__()
        # Initialize necessary layers and configurations
        self.add_layer = tf.keras.layers.Add()
        self.rescaling = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)
        self.layer = tf.keras.layers.Layer()
        self.string_lookup = tf.keras.layers.experimental.preprocessing.StringLookup()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='linear', kernel_initializer=tf.initializers.truncated_normal())
        self.used_apis = [
            'tf.keras.layers.experimental.preprocessing.Rescaling',
            'tf.keras.layers.Layer',
            'tf.keras.layers.Dense',
            'tf.clip_by_norm',
            'tf.pad',
            'tf.experimental.numpy.dstack',
            'tf.experimental.numpy.sort',
            'tf.experimental.numpy.exp2',
            'tf.signal.ifftshift',
            'tf.gather_nd'
        ]

    def call(self, inputs):
        # Record input-output of each API
        api_outputs = {}

        api_outputs['pre_self.rescaling(inputs)'] = inputs
        x = self.rescaling(inputs)
        api_outputs['self.rescaling(inputs)'] = x

        # Flatten the input
        api_outputs['pre_self.flatten(x)'] = x
        x = self.flatten(x)
        api_outputs['self.flatten(x)'] = x

        # Using tf.keras.layers.Dense
        api_outputs['pre_self.dense(x)'] = x
        x = self.dense(x)
        api_outputs['self.dense(x)'] = x

        # Additional API usages from the list
        api_outputs['pre_tf.clip_by_norm(x, 1.0)'] = x
        x = tf.clip_by_norm(x, 1.0)
        api_outputs['tf.clip_by_norm(x, 1.0)'] = x

        api_outputs['pre_tf.pad(x, paddings=[[0, 0], [0, 1]], mode="CONSTANT")'] = x
        x = tf.pad(x, paddings=[[0, 0], [0, 1]], mode="CONSTANT")
        api_outputs['tf.pad(x, paddings=[[0, 0], [0, 1]], mode="CONSTANT")'] = x

        api_outputs['pre_tf.experimental.numpy.dstack([x, x])'] = x
        x = tf.experimental.numpy.dstack([x, x])
        api_outputs['tf.experimental.numpy.dstack([x, x])'] = x

        api_outputs['pre_tf.experimental.numpy.sort(x)'] = x
        x = tf.experimental.numpy.sort(x)
        api_outputs['tf.experimental.numpy.sort(x)'] = x

        api_outputs['pre_tf.experimental.numpy.exp2(x)'] = x
        x = tf.experimental.numpy.exp2(x)
        api_outputs['tf.experimental.numpy.exp2(x)'] = x

        api_outputs['pre_tf.signal.ifftshift(x)'] = x
        x = tf.signal.ifftshift(x)
        api_outputs['tf.signal.ifftshift(x)'] = x
        # Final output calculation

        api_outputs['pre_tf.gather_nd'] = x
        output = tf.gather_nd(x, tf.ragged.constant([[[0], [0]]], ragged_rank=1), batch_dims=1)
        api_outputs['tf.gather_nd'] = x

        # List of used APIs
        used_apis_sorted = sorted(set(self.used_apis), key=self.used_apis.index)

        return output, used_apis_sorted