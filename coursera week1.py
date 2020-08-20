import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([1,2,3,4,5,6,8,10,11,14], dtype=float)
ys = np.array([100,150,200,250,300,350,450,550,600,750],dtype = float)

model.fit(xs,ys,epochs = 500)
print(model.predict([7]))