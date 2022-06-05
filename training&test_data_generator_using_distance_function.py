import numpy as np 

X_train = np.random.randint(0, 10, size=(10000, 10))

X_test = np.random.randint(0, 10, size=(100, 10))

y_train = np.array([1 if np.sqrt((X_train*X_train).sum(axis=1))[i]<=15 else 0 for i in range(0, 10000)]) #use various other values besides 15, since,
#in that way, the same data set can be the input for different concepts (captured by the values of y_train/test) which may both be still true if, say, for d=15 is true and, then, if the
#other concept has d=20, it will also be true. In particular, this is an implication (it can also be used to validate the conjuction of the concepts involved).

y_test = np.array([1 if np.sqrt((X_test*X_test).sum(axis=1))[i]<=15 else 0 for i in range(0, 100)])

np.savetxt("X_train.csv", X_train, delimiter=",")

np.savetxt("X_test.csv", X_test, delimiter=",")

np.savetxt("y_train.csv", y_train, delimiter=",")

np.savetxt("y_test.csv", y_test, delimiter=",")

#print (X_train)

#print (np.sqrt ((X_train*X_train).sum(axis=1)))

#print (y_train)

#print (X_train.shape)

#print (y_train.shape)








