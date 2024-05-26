import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, losses

__all__ = [
    'main_voc',
    'main_train',
    'main_retrain',
    'train',
    'show_history',
    'values',
    'examples',
    'predict_model',
]


def main_voc(ds_path, text_layer, voc_path):
    df = pd.read_excel(ds_path)

    text_layer.adapt(df['name'])

    voc = text_layer.get_vocabulary()

    with open(voc_path, 'w') as f:
        f.write(json.dumps(voc))


def main_train(ds_path, voc_path, text_layer, model, model_path):
    with open(voc_path, 'r') as f:
        voc = json.loads(f.read())

    text_layer.set_vocabulary(voc)

    df = pd.read_excel(ds_path)
    ds = tf.data.Dataset.from_tensor_slices((
        df['name'],
        df[['yes', 'no']]
    ))

    train(ds, text_layer, model)

    model.save(model_path)


def main_retrain(ds_path, voc_path, text_layer, model_path):
    with open(voc_path, 'r') as f:
        voc = json.loads(f.read())
    text_layer.set_vocabulary(voc)

    model = tf.keras.models.load_model(model_path)

    print('First predict')
    predict_model(model, text_layer)
    print()

    print('Re-train')
    df = pd.read_excel(ds_path)
    for name, yes, no in values:
        df = df._append(
            dict(name=name, yes=yes, no=no),
            ignore_index=True
        )

    ds = tf.data.Dataset.from_tensor_slices((
        df['name'],
        df[['yes', 'no']],
    ))

    train(ds, text_layer, model, adapt=True)


def train(ds, text_layer, model, adapt=False):
    batch_size = 32
    ds = ds.shuffle(len(ds)).batch(batch_size)

    cnt = len(ds)
    train_cnt = int(cnt * 0.8)
    val_cnt = int(cnt * 0.1)
    test_cnt = int(cnt * 0.1)

    raw_train_ds = ds
    raw_val_ds = ds.shard(10, 1)
    raw_test_ds = ds.shard(10, 2)

    print(f'Batch size: {batch_size}; '
          f'Train: {len(raw_train_ds)}; '
          f'Val: {len(raw_val_ds)}; '
          f'Test: {len(raw_test_ds)}')

    if adapt:
        text_layer.adapt(raw_train_ds.map(lambda x, _: x))

    def text_vectorize(features, labels):
        features = tf.expand_dims(features, axis=-1)
        return text_layer(features), labels

    train_ds = raw_train_ds.map(text_vectorize)
    val_ds = raw_val_ds.map(text_vectorize)
    test_ds = raw_test_ds.map(text_vectorize)

    model.compile(
        loss=losses.BinaryCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['binary_accuracy'],
    )

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )

    show_history(history)

    predict_model(model, text_layer)

    return model


def show_history(history):
    history_dict = history.history

    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()


values = [
    ('The movie was great!', 0, 1),
    ('The movie was okay.', 0, 1),
    ('The movie was terrible..', 0, 1),
    ('PyCharm', 0, 1),
    ('Kaspersky Endpoint Security', 1, 0),
    ('McAfee', 1, 0),
    ('McAfee antivirus', 1, 0),
    ('Dr.Web', 1, 0),
    ('Dr.Web antivirus', 1, 0),
]
examples = [name for name, _, _ in values]
examples = np.array(examples).astype(object)


def predict_model(model, text_layer):
    model = tf.keras.Sequential([
        text_layer,
        model,
        layers.Activation('sigmoid')
    ])
    model.compile(
        loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )
    predict = model.predict(examples)
    _width = max(map(len, examples)) + 2
    print('\n'.join(
        f'{e:<{_width}}: {yes = :.2f} {no = :.2f}'
        for e, (yes, no) in zip(examples, predict)
    ))
