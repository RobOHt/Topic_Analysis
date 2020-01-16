import keras
from keras.datasets import reuters
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Embedding, Conv1D, GlobalMaxPooling1D
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
(x_train, Y_train), (x_test, Y_test) = reuters.load_data(num_words=None, test_split=0.2)
np.load = np_load_old

num_classes = max(Y_train) + 1

max_words = 100
embedding_dim = 256

tokeniser = Tokenizer(num_words=max_words, split=" ")

x_train = tokeniser.sequences_to_matrix(x_train, mode='count')
x_train = pad_sequences(x_train)
x_test = tokeniser.sequences_to_matrix(x_test, mode='count')
x_test = pad_sequences(x_test)

print(x_train)

Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

# print(x_train[0])

model = Sequential()

model.add(Embedding(len(x_train), 256, input_length=x_train.shape[1]))
model.add(Dropout(0.2))

model.add(LSTM(256, activation='relu', return_sequences=True, dropout=0.3, recurrent_dropout=0.2))

model.add(LSTM(256, activation='relu', dropout=0.3, recurrent_dropout=0.2))

model.add(Dense(num_classes, activation='softmax'))

optimiser = keras.optimizers.Adam(lr=1e-3, decay=1e-20)
model.compile(optimizer=optimiser,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, Y_train, batch_size=32, epochs=3, validation_split=0.3)
# score = model.evaluate(x_test, Y_test, batch_size=32)

# print(score)

