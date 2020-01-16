import keras
from keras.datasets import reuters
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Embedding, Conv1D, GlobalMaxPooling1D
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb

max_features = 5000
maxlen = 200
batch_size = 64
embedding_dims = 16
filters = 128
kernal_size = 3
hidden_size = 128
epochs = 2

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
(x_train, Y_train), (_, _) = imdb.load_data(num_words=max_features)
np.load = np_load_old

x_train = pad_sequences(x_train, maxlen=maxlen)
print(x_train[0])

model = Sequential()

model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(Conv1D(filters, kernal_size, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train, Y_train, batch_size=batch_size, epochs=epochs)

