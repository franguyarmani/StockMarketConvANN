import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling1D, UpSampling1D
from tensorflow.keras.layers import Input, concatenate, Conv1D, LeakyReLU, AveragePooling1D, Average
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import datetime
import pandas
import os
from os.path import isfile, join, splitext

import Utilities as util


def aggregate_ticker_data(output_name, content_directory, start_date, end_date, col_id='% change'):
    os.chdir(content_directory)
    file_names = [f for f in os.listdir(os.getcwd()) if isfile(join(os.getcwd(), f))]
    aggregated = pandas.DataFrame()

    i = 0
    for name in file_names:
        if i % 100 == 0:
            print(i)
        file = pandas.read_csv(name)
        file = file.set_index('Date')
        if file.first_valid_index() <= start_date:
            util.calculate_changes(file)
            aggregated[name] = file["% change"][start_date:end_date]
        i += 1
    os.chdir('../')
    aggregated.to_csv(output_name)
    return aggregated


aggregate_df = aggregate_ticker_data('FirstTest.csv', 'stock-market-dataset/stocks', '2000-01-01', '2020-01-01')
