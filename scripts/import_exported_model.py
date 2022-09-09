import os

import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *

if __name__ == '__main__':
    # Audio Test File
    path_to_test_audio = f"../test_data/{os.listdir('../test_data/')[1]}"
    wav_test_data = load_wav_16k_mono(path_to_test_audio)

    my_classes = ['dog', 'cat']

    # Path to Model Export
    model_path = '../brand_new_yamnet'

    # Inference on Exported Model
    reloaded_model = tf.saved_model.load(model_path)
    reloaded_results = reloaded_model(wav_test_data)
    cat_or_dog = my_classes[tf.argmax(reloaded_results)]
    print(f'The main sound is: {cat_or_dog}')