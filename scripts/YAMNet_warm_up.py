import os

import numpy as np
import pandas as pd

import tensorflow as tf
from YAMNet_audio_detection import YAMNet_is_listening
import tensorflow_io as tfio
from utils import load_wav_16k_mono
import tensorflow_hub as hub

class YAMNet_goes_to_gym:
    def __init__(self, path_to_csv_data, path_to_audio):
        self.path_to_audio = path_to_audio
        self.path_to_csv_data = path_to_csv_data

    # Model returns 3 arguments: scores, embeddings and spectogram
    def load_yamnet(self):
        yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        yamnet_model = hub.load(yamnet_model_handle)
        return yamnet_model

    def prepare_data(self):
        my_classes = ['dog', 'cat']
        map_class_to_id = {'dog':0, 'cat':1}

        self.csv_data = pd.read_csv(self.path_to_csv_data)

        filtered_pd = self.csv_data[self.csv_data.category.isin(my_classes)]

        class_id = filtered_pd['category'].apply(lambda name: map_class_to_id[name])
        filtered_pd = filtered_pd.assign(target=class_id)

        full_path = filtered_pd['filename'].apply(lambda row: os.path.join(self.path_to_audio, row))
        filtered_pd = filtered_pd.assign(filename=full_path)
        
        data = filtered_pd[['filename','target']]
        filename = filtered_pd['filename']
        labels = filtered_pd['target']
        return data, filename, labels, my_classes
    
    def convert_data_to_tensor_ds(self, filename, label):
        main_ds = tf.data.Dataset.from_tensor_slices((filename, label))
        return main_ds

    def load_wav_for_map(self, filename, label):
        return load_wav_16k_mono(filename), label

    # YAMNet as an Embeddings Layer
    def extract_embedding(self, wav, label):
        ''' run YAMNet to extract embedding from the wav data '''
        yamnet_model = self.load_yamnet()
        scores, embeddings, spectrogram = yamnet_model(wav)
        num_embeddings = tf.shape(embeddings)[0]
        return (embeddings, tf.repeat(label, num_embeddings))