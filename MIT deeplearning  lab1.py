import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as plt

sports = tf.constant("tennis", tf.string)
number = tf.constant(1.1, tf.float64)

print("`sport` is a {}-d Tensor".format(tf.rank(sports).numpy()))
print("`number` is a {}-d Tensor".format(tf.rank(number).numpy()))

sports = tf.constant(["tennis"], tf.string)
number = tf.constant([1.1], tf.float64)

print(" sport is a {}-d Tenosr".format(tf.rank(sports).numpy()))
print(" number is a {}-d Tenosr".format(tf.rank(number).numpy()))

sports = tf.constant([[[["tennis"], ["football"]]], [[["bsketball"], ["baseball"]]]], tf.string)
number = tf.constant([[[[1.1], [1.2]]], [[[1.3], [1.5]]]], tf.float64)

print(" sport is a {}-d Tenosr".format(tf.rank(sports).numpy()))
print(" number is a {}-d Tenosr".format(tf.rank(number).numpy()))

matrix = tf.zeros([10, 256, 256, 3])

print(matrix)

row_vector = matrix[1]
column_vector = matrix[:, 2]
scalar = matrix[1, 2]

print("row_vector:{}".format(row_vector.numpy()))
print("column_vector:{}".format(column_vector.numpy()))
print("scalar:{}".format(scalar.numpy()))

a = tf.constant(15)
b = tf.constant(15)
c = tf.constant(10)

c1 = tf.add(a, c)
c2 = a + b

print(c1)


def func(a, b):
    c = tf.add(a, b)
    d = tf.subtract(b, 1)
    e = tf.multiply(d, c)
    return e


print(func(3, 2))


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, n_output_nodes):
        super(DenseLayer, self).__init__()
        self.n_output_nodes = n_output_nodes

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.w = self.add_weight("weight", shape=[d, self.n_output_nodes])
        self.b = self.add_weight("bias", shape=[1, self.n_output_nodes])

    def call(self, x):
        z = tf.matmul(x, self.w) + self.b

        y = tf.sigmoid(z)
        return y


tf.random.set_seed(1)
layer = DenseLayer(3)
layer.build((1, 2))
x_input = tf.constant([[1, 2.]], shape=(1, 2))
y = layer.call(x_input)

print(y.numpy())


n_output_nodes = 3
model = keras.Sequential()
dense_layer = keras.layers.Dense(n_output_nodes, activation='sigmoid')
model.add(dense_layer)

model_output = model(x_input).numpy()
print(model_output)