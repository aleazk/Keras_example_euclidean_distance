import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import keras 

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

X_train, y_train = np.loadtxt("X_train.csv", delimiter=","), np.loadtxt("y_train.csv", delimiter=",")

X_test, y_test = np.loadtxt("X_test.csv", delimiter=","), np.loadtxt("y_test.csv", delimiter=",")

#print (X_train.shape)

num = X_train.shape[1]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

def classification_model():
	model = Sequential()
	model.add(Dense(num, activation="relu", input_shape=(num,)))
	model.add(Dense(100, activation="relu"))
	model.add(Dense(num_classes, activation="softmax"))
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
	return model

model = classification_model()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)

print ("Accuracy: {}%\n Error: {}".format(scores[1], 1 - scores[1]))

model.save("classification_model.h5")


