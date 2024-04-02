# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 2024

@author: Ethan
"""

# For PostgreSQL and the Neural Network
import psycopg2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.utils import normalize

# Disabling Tensorflow Warnings
import tensorflow as tf
tf.autograph.set_verbosity(0)

#establishing the connection
conn = psycopg2.connect(
   database='iris', user='postgres', password='------', host='127.0.0.1', port='5432'
)
cursor = conn.cursor()

# Initialize dataset size and label array
num_vals = 150
y = np.empty((num_vals, 3)) # labels

# Fetching Inputs
cursor.execute(f"select sepal_length, sepal_width, petal_length, petal_width from iris_dataset limit {num_vals}")
data = cursor.fetchall()
X = np.array(data)

# Normalize
X = normalize(X, axis=0)

# Fetching Labels
cursor.execute(f"select species from iris_dataset limit {num_vals}")
species = cursor.fetchall()
for i in range(len(y)):
    for i in range(len(y)):
        if (species[i][0] == 'setosa'):
            y[i] = [1, 0, 0]
        elif (species[i][0] == 'versicolor'):
            y[i] = [0, 1, 0]
        else: # 'virginica'
            y[i] = [0, 0, 1]

#Closing the connection
conn.close()
print('Training Model...')

# Split data into Training/Validation and Testing
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.05, random_state=1)

# Split Training/Validation into separate lists
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.16, random_state=1)

# Define Keras model
model = Sequential()
model.add(Dense(16, input_shape=(4,), activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

# Compile Keras model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=300, batch_size=16, verbose=0, validation_data=(X_val, y_val))

loss, accuracy = model.evaluate(X_test, y_test)
print('Loss: %.2f | Accuracy: %.2f' % (loss*100, accuracy*100))
print()

# Predicting with the model

print('Testing Model...')
predictions = model.predict(X_test)

for i in range(len(X_test)):
    print(f'X_test[{i}] => {predictions[i]} (expected {y_test[i]})')
