import os

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio


# Utility functions for loading audio files and making sure the sample rate is correct.
@tf.function
def load_wav_16k_mono(path_to_audio):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    frequency = 16000
    file_contents = tf.io.read_file(path_to_audio)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=frequency)
    return wav