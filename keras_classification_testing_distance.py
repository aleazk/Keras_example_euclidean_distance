import numpy as np 


import keras 


from keras.utils import to_categorical

X_train, y_train = np.loadtxt("X_train.csv", delimiter=","), np.loadtxt("y_train.csv", delimiter=",")

X_test, y_test = np.loadtxt("X_test.csv", delimiter=","), np.loadtxt("y_test.csv", delimiter=",")

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


from keras.models import load_model

pretrained_model = load_model("classification_model.h5")

predictions = pretrained_model.predict(X_test[0:3])

print (predictions)

print (y_test[0:3])
