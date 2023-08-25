import os

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import tensorflow as tf
from numpy.core.records import ndarray
from statsmodels.tsa.api import Holt


def reshape_ts(ts, label):
    """
        To input a time series into a conv net, its shape needs to be (num_points, 1, 1). Upon loading from dataset, the shape is just (num_points,). Thus, the time series needs to be reshaped with this function
    """
    return tf.reshape(ts, (ts.shape[0], 1)), label


def _read_raw_data(
        data_dir='static/time_synth/out',
        data_file='dataset.csv', labels_file='labels.csv', 
        labels_column='name') -> tuple[ndarray, pd.DataFrame, list[str]]:
    """
        Function loads time series data, and respective labels

        Args :
            - data_dir (string): directory in which the data and labels are located
        
        Returns :
            - ts_points (NumPy Array) : Array of shape (num_series, num_points). Contains point data of the time series
            - labels (Pandas DataFrame) : Contains three one-hot encoded values: type of pattern (column name: 'name'), its position in the series (column name: 'position'), and its duration (column name: 'width').
            - classes (list[string]) : The names of the one-hot encoded categorical classes in their data-frame order
    """
    ts_points = pd.read_csv(os.path.join(data_dir, data_file), header=None).to_numpy()
    labels = pd.read_csv(os.path.join(data_dir, labels_file))
    classes = sorted(labels[labels_column].str.lower().unique())

    labels[labels_column] = labels[labels_column].str.lower()
    labels = pd.get_dummies(labels, columns=[labels_column], prefix='', prefix_sep='')

    return ts_points, labels, classes


def load_dataset(
        data_dir='static/time_synth/out',
        data_file='dataset.csv', 
        labels_file='labels.csv', drop_columns=[]) -> tuple[tf.data.Dataset, list[str]]:
    """
        Function returns the time series dataset

        Args :
            - data_dir (string): directory in which the data and labels are located
            - drop_columns (list[string]) : list of column names from the dataset to be dropped. Possible values are any or all of the following: 'name', 'position', 'width'
        Returns :
            - dataset (TensorFlow Dataset) : dataset containing data and values
            - classes (list[string]) : The names of the one-hot encoded categorical classes in their data-frame order
    """
    ts_points, labels, classes = _read_raw_data(data_dir, data_file, labels_file)
    if drop_columns:
        labels = labels.drop(columns=drop_columns)

    dataset = tf.data.Dataset.from_tensor_slices((ts_points, labels))
    dataset = dataset.map(reshape_ts, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset, classes


def split(dataset: tf.data.Dataset, train=0.8, val=0.1, test=0.1) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
        Splits the dataset into train, validation and test splits. It shuffles the dataset before the splitting, so each time this function is called, it returns differently ordered dataset

        Args :
            - dataset (tensorflow Dataset) : the dataset to be split
            - train, test, val (float) : percentages of the original dataset to be used for each respective split

        Returns :
            - train, validation and test datasets (tensorflow Dataset) : the splits of the dataset

        Errors :
            - ValueError : The function throws a ValueError if the proportions of the train, val and test datasets do not add up to 1, or if they are negative numbers  
    """
    # because of floating point arithmetic, the sum is never exactly 1.0
    if not 0.999 < train + test + val < 1.001:
        raise ValueError('Train, test and val must add up to 100%')
    if train < 0 or test < 0 or val < 0:
        raise ValueError('Train, test and validation must be positive numbers between 0 and 1')

    buffer_size = int(len(dataset))
    val_size = int(val * buffer_size)
    test_size = int(test * buffer_size)

    dataset = dataset.shuffle(buffer_size)  # shuffle for randomness for each run
    val_dataset = dataset.take(val_size)
    test_dataset = dataset.skip(val_size).take(test_size)
    train_dataset = dataset.skip(val_size+test_size)

    return train_dataset, val_dataset, test_dataset


def decode_one_hot(one_hot, classes):
    return classes[tf.argmax(one_hot, axis=1)[0]]


def windowed_dataset(series, window_size, batch_size,
                     normalize_window=True, labeled=False, smoothing=None, shift=1) -> tf.data.Dataset:
    """Generates dataset windows

    Args:
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to include in the feature
      batch_size (int) - the batch size
      normalize_window (int) - the batch
      labeled (bool) - default False. If the value is set to true, the labels are the same as data. used for
      model pretraining, especially when using auto-encoders
      smoothing (string) - if labeled is set to true, this sets the smoothing function to be used. Possible values are
      'holt', 'rolling', None (default)
      shift (int) - The shift argument determines the number of input elements to shift between the start of each window

    Returns:
      dataset (TF Dataset) - TF Dataset containing time windows
    """
    def normalize(window):
        return (window - tf.reduce_min(window)) / (tf.reduce_max(window) - tf.reduce_min(window))

    def normalize_expand(window, normalized):
        if normalized:
            window = normalize(window)
        return tf.expand_dims(window, axis=-1)

    def holt_smoothing(time_series):
        return Holt(time_series).fit(smoothing_level=0.3, smoothing_trend=0.05, optimized=False).fittedvalues

    def rolling_average_smoothing(time_series):
        kernel = np.ones(16) / 16
        smoothed_series = np.convolve(time_series, kernel, mode='same')
        return smoothed_series

    def idf(time_series):
        return time_series

    smoothing_functions = {
        'none': idf,
        'holt': holt_smoothing,
        'rolling': rolling_average_smoothing
    }

    smoothing_function = idf if smoothing is None else smoothing_functions[smoothing]
  
    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size, shift=shift, drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda window: window.batch(window_size))

    dataset = dataset.map(lambda window: normalize_expand(window, normalize_window))

    if labeled:
        smooth_series = smoothing_function(series)
        labels_dataset = tf.data.Dataset.from_tensor_slices(smooth_series)
        labels_dataset = labels_dataset.window(window_size, shift=shift, drop_remainder=True)
        labels_dataset = labels_dataset.flat_map(lambda window: window.batch(window_size))
        labels_dataset = labels_dataset.map(lambda window: normalize_expand(window, normalize_window))
        dataset = tf.data.Dataset.zip((dataset, labels_dataset))

    # Create batches of windows
    dataset = dataset.batch(batch_size)
    # Prefetch optimization
    dataset = dataset.prefetch(batch_size)
    
    return dataset


if __name__ == "__main__":
    data, classes = load_dataset()
    print(classes)
    train, test, val = split(data)
    print(len(train), len(test), len(val))

    for i, element in enumerate(data.as_numpy_iterator()):
        if i > 5:
            break
        print(element[0].shape)
        print(element[1])
