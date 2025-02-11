# -*- coding: utf-8 -*-
"""Churn_modelling.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HsCv8gvD4lebMlWBkDUvPwxQIUJA5Zv3
"""

!pip install tensorflow-gpu

import tensorflow as tf
print(tf.__version__)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LeakyReLU, PReLU, ELU

df = pd.read_csv('Churn_Modelling.csv')

df.head()

X = df.iloc[:,3:13]

X.head()

y = df.iloc[:,-1]

y.head()

geography = pd.get_dummies(X['Geography'], drop_first=True).astype(int)

gender = pd.get_dummies(X['Gender']).astype(int)
gender.head(
)

X = X.drop(['Geography', 'Gender'], axis = 1)

df.head()

X.head()

pd.concat([ X, geography, gender] , axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


classifier = Sequential()

classifier.add(Dense(units = 11, activation = 'relu'))

classifier.add(Dense(units=7, activation='relu'))

classifier.add(Dense(units = 8, activation='relu'))

classifier.add(Dense(1, activation='sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=0,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)

model = classifier.fit(X_train, y_train, validation_split = 0.33, batch_size = 10, epochs = 100, callbacks = [early_stop])

y_pred = classifier.predict(X_test)

y_pred = (y_pred >= 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)





