import numpy as np
from scipy.signal import windows
import matplotlib.pyplot as plt
import pywt
from scipy.io import wavfile
import os
from shutil import rmtree, copy
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense, Input, Activation, BatchNormalization, Dropout
from tensorflow.keras.optimizers import SGD
import datetime
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import random
import librosa
import scipy.stats
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import sklearn.metrics