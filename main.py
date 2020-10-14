import pandas
from tensorflow.python.keras.optimizers import Adam

import DataGenerator
import StockPredictionModel

data_path = '/content/stocks'
x_col = 'High'
y_col = '% change'
x_len = 16
y_len = 8 # y-
batch_size = 32
opt = Adam(lr=0.0001)

x_data = pandas.read_csv('train_data.csv')
cutoff_percent = 0.1
ticker = 'AAPL'


x_data = x_data.drop(columns='Unnamed: 0')

data_generator = DataGenerator(x_data, x_len, y_len, cutoff_percent, batch_size, ticker)

train_gen = data_generator.generator()

example_x, example_y = train_gen.next()
input_shape = (example_x.shape[1], example_x.shape[2])


model = StockPredictionModel(input_shape, 15)
model.build_model()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# log the experiment progress to comet ml, view progress here: https://www.comet.ml/cm5409a/stock-market-cnn

num_epochs = 100

# steps_per_epoch = y_train.shape[0] // batch_size

model.fit(train_gen,
          steps_per_epoch = 1000, # some random number for now
          epochs=num_epochs)
