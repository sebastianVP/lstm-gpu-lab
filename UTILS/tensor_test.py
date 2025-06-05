"""
TensorFlow
"""

import tensorflow as tf

print("TensorFlow version:",tf.__version__)
print("GPU available:",len(tf.config.list_physical_devices("GPU"))>0)
print("GPUs:",tf.config.list_physical_devices("GPU"))
