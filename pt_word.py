import tensorflow as tf


def text_prep(input_value):
    input_value = tf.strings.lower(input_value)
    input_value = tf.strings.regex_replace(input_value, r'[^a-zа-яё]+', ' ')
    input_value = tf.strings.regex_replace(input_value, ' +', ' ')
    return input_value


max_features = 10 ** 6
sequence_length = 50

text_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_features,
    standardize=text_prep,
    output_mode='int',
    output_sequence_length=sequence_length,
)
