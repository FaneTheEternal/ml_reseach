import sys

import tensorflow as tf
from tensorflow.keras import layers

from pt_char2 import text_layer, max_features
from pt_utils import main_train

ds_file, voc_file, model_file = sys.argv[1:]

embedding_dim = 32
model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Conv1D(256, 3),
    layers.GlobalMaxPool1D(),
    layers.Dense(2),
])

main_train(ds_file, voc_file, text_layer, model, model_file)
