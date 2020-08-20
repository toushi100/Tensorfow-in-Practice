import tensorflow as tf
import numpy as np
from tensorflow import keras


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > 0.95):
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
callback = myCallback()
train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images / 255
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255

model = tf.keras.models.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=3, callbacks=[callback])
print("model evaluation")
model.evaluate(test_images, test_labels)