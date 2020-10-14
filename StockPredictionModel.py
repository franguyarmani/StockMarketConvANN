from tensorflow.keras.layers import Conv1D, Input, Flatten, Dense
from tensorflow.keras.models import Sequential


class StockPredictionModel():
    def __init__(self, input_shape, kernel_size, activation='sigmoid'):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.activation = 'sigmoid'
        self.model = Sequential()

    def build_model(self):
        self.model.add(Input(shape=self.input_shape))
        self.model.add(Conv1D(256, self.kernel_size, padding='same', activation=self.activation))
        self.model.add(Conv1D(256, self.kernel_size, padding='same', activation=self.activation))
        self.model.add(Conv1D(128, self.kernel_size, padding='same', activation=self.activation))
        self.model.add(Flatten())
        self.model.add(Dense(3, activation='softmax'))
        return self.model
