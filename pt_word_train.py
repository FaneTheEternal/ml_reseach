import sys

import tensorflow as tf
from tensorflow.keras import layers

from pt_utils import *
from pt_word import text_layer, max_features

ds_file, voc_file, model_file = sys.argv[1:]

embedding_dim = 16
model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(2),
])

main_train(ds_file, voc_file, text_layer, model, model_file)
