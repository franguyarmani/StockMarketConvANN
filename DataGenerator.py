from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import Utilities as util

# cutoff_percent = where
# new data generator

# cutoff_percent = where
# new data generator
class DataGenerator():
  def __init__(self, x_data, x_len, y_len, cutoff_percent, batch_size, ticker):
    self.x_data = x_data
    self.batch_size = batch_size
    self.x_len = x_len
    self.y_len = y_len
    self.num_examples = x_data.shape[0]
    self.cutoff_percent = cutoff_percent
    self.ticker = ticker

  def generate_y(self, window):
    net_change = util.calc_net_change(window)

    label = 1
    if net_change > self.cutoff_percent:
      label = 2
    elif net_change < -self.cutoff_percent:
      label = 0

    one_hot_encoded = to_categorical(label, num_classes=3)
    return one_hot_encoded

  def generator(self):
    while True:
      x = []
      y = []

      for i in range(self.batch_size):
        rand_idx = np.random.randint(0, high=self.num_examples - self.x_len - 1)

        this_x = self.x_data[rand_idx: rand_idx + self.x_len]

        window = self.x_data[rand_idx + self.x_len: rand_idx + self.x_len + self.y_len]

        this_y = self.generate_y(window[self.ticker].to_numpy())

        this_x = this_x.to_numpy()

        x.append(this_x)
        y.append(this_y)

      x = np.asarray(x)
      y = np.asarray(y)

      y = tf.reshape(y, [y.shape[0], y.shape[1], 1])
      x = tf.cast(x, dtype=tf.float32)
      y = tf.cast(y, dtype=tf.float32)
      yield x, y