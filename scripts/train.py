import os

import numpy as np
import pandas as pd

import tensorflow as tf
from YAMNet_audio_detection import YAMNet_is_listening
import tensorflow_io as tfio
import tensorflow_hub as hub
from YAMNet_warm_up import YAMNet_goes_to_gym
from utils import *

# Enables to get Audio data as an Input for the Exported Model
class ReduceMeanLayer(tf.keras.layers.Layer):
  def __init__(self, axis=0, **kwargs):
    super(ReduceMeanLayer, self).__init__(**kwargs)
    self.axis = axis

  def call(self, input):
    return tf.math.reduce_mean(input, axis=self.axis)

# Train, Test and Val Split
def split_ds(ds, batch_size):
    ds_len = ds.reduce(0, lambda x,_: x+1).numpy()
    train_size = int(0.7 * ds_len)
    val_size = int(0.17 * ds_len)
    test_size = int(0.13 * ds_len)

    train_ds = ds.take(train_size)
    test_ds = ds.skip(train_size)
    val_ds = test_ds.skip(val_size)
    test_ds = test_ds.take(test_size)

    train_ds = train_ds.cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds

# Model Architecture
def model_architecture(labels):
    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024), dtype=tf.float32, name='input_embedding'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(labels)))
    ], name='custom_model')

    print(model.summary())

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer="adam",
                 metrics=['accuracy'])

    return model

# Fit
def custom_model_fit(model, labels, train_ds, val_ds):
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=3,
                                            restore_best_weights=True)

    history = model.fit(train_ds,
                    epochs=20,
                    validation_data=val_ds,
                    callbacks=callback)

    return history

# Evaluation
def model_evaluation(model, test_ds):
    loss, accuracy = model.evaluate(test_ds)

    return loss, accuracy

# Export
def export_model(yamnet_model, custom_model, model_path):
    input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
    embedding_extraction_layer = hub.KerasLayer(yamnet_model, trainable=False, name='yamnet')
    _, embeddings_output, _ = embedding_extraction_layer(input_segment)
    serving_outputs = custom_model(embeddings_output)
    serving_outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)
    serving_model = tf.keras.Model(input_segment, serving_outputs)
    serving_model.save(model_path, include_optimizer=False)

if __name__ =='__main__':
    BATCH_SIZE = 16
    # Get Audio and Metadata
    path_to_csv_data = '../ESC-50/meta/esc50.csv'
    path_to_audio = '../ESC-50/audio'

    # Audio Test File
    path_to_test_audio = f"../test_data/{os.listdir('../test_data/')[0]}"

    # Path to Model Export
    model_path = '../brand_new_yamnet'

    ### Training Phase
    # YAMNet Training Preparation
    train_yamnet = YAMNet_goes_to_gym(path_to_csv_data, path_to_audio)
    yamnet_model = train_yamnet.load_yamnet()
    # wav_test_data = load_wav_16k_mono(path_to_test_audio)

    data, filename, labels, my_classes = train_yamnet.prepare_data()
    ds = train_yamnet.convert_data_to_tensor_ds(filename, labels)

    main_ds = ds.map(train_yamnet.load_wav_for_map)
    main_ds = main_ds.map(train_yamnet.extract_embedding).unbatch()

    # Train, Test and Val Split
    train_ds, val_ds, test_ds = split_ds(main_ds, BATCH_SIZE)

    # Custom Model Architecture
    custom_model = model_architecture(labels)

    # Fit
    history = custom_model_fit(custom_model, labels, train_ds, val_ds)

    # Evaluate
    loss, accuracy = model_evaluation(custom_model, test_ds)

    # Inference After Train (Optional)
    # scores, embeddings, spectrogram = yamnet_model(wav_test_data)
    # result = custom_model(embeddings).numpy()

    # inferred_class = my_classes[result.mean(axis=0).argmax()]
    # print(f'The main sound is: {inferred_class}')

    # Export Trained Model
    export_model(yamnet_model, custom_model, model_path)