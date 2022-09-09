import os
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
from utils import load_wav_16k_mono

class YAMNet_is_listening:
    def __init__(self, path_to_audio):
        self.path_to_audio = path_to_audio

    # Model returns 3 arguments: scores, embeddings and spectogram
    def load_yamnet(self):
        yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        self.yamnet_model = hub.load(yamnet_model_handle)
        return self

    # Load YAMNet class labels
    def load_labels(self):
        class_map_path = self.yamnet_model.class_map_path().numpy().decode('utf-8')
        class_names = list(pd.read_csv(class_map_path)['display_name'])
        return class_names

    ### YAMNet Provides pre-trained frame-level class scores.
    # Inference with YAMNet embeddings
    def yamnet_predict(self):
        wav = load_wav_16k_mono(self.path_to_audio)
        # Embeddings Shape -> (N, 1024); Where N is the number of frames of YAMNet found (every 0.48 seconds of audio)
        scores, embeddings, spectrogram = self.yamnet_model(wav)
        return scores, embeddings, spectrogram

    def predict_class_and_confidence(self, scores, labels):
        # scores aggregation per class across frames
        class_scores = tf.reduce_mean(scores, axis=0)
        
        # get predicted class
        top_class = tf.argmax(class_scores)
        inferred_class = labels[top_class]
        
        # get confidence score for the predicted class
        confidence = class_scores[top_class].numpy()

        return inferred_class, confidence

    def plot_results(self, scores, labels, spectrogram, wav_data):

        top_n = 5

        mean_scores = np.mean(scores, axis=0)
        top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
        yticks = range(0, top_n, 1)
        patch_padding = (0.025 / 2) / 0.01

        plt.plot(wav_data)
        plt.xlim([0, len(wav_data)])
        plt.savefig("../visualizations/waveform_noise_reduction.png")
        plt.show()

        plt.imshow(spectrogram.numpy().T, aspect='auto', interpolation='nearest', origin='lower')
        plt.savefig("../visualizations/yamnet_spectogram.png")
        plt.show()

        plt.xlim([-patch_padding-0.5, scores.shape[0] + patch_padding-0.5])
        plt.yticks(yticks, [labels[top_class_indices[x]] for x in yticks])
        plt.imshow(scores.numpy()[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')
        plt.savefig("../visualizations/yamnet_plot.png")
        plt.show()

        plt.plot(scores.numpy()[:, top_class_indices])
        plt.legend([labels[index] for index in top_class_indices])
        plt.xlabel('Frames')
        plt.ylabel('Probability')
        plt.ylim(0, 1.)
        plt.savefig("../visualizations/noise.png")
        plt.show()