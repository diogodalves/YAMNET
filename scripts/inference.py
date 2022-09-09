from tkinter import S
import tensorflow as tf
import numpy as np
import pandas as pd
from YAMNet_audio_detection import YAMNet_is_listening
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':

    path_to_audio = f"../test_data/{os.listdir('../test_data/')[1]}"

    yamnet = YAMNet_is_listening(path_to_audio)
    model = yamnet.load_yamnet()
    labels = yamnet.load_labels()
    scores, embeddings, spectrogram = yamnet.yamnet_predict()
    label, confidence = yamnet.predict_class_and_confidence(scores, labels)