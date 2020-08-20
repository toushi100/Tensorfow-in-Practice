import tensorflow as tf
import numpy as np
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import os

base_dir = 'tmp/cats_and_dogs_filtered'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(base_dir,
                                                    target_size=(300, 300),
                                                    batch_size=120,
                                                    class_mode="binary")

validate_datagen = ImageDataGenerator(rescale=1. / 255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')
validate_generator = validate_datagen.flow_from_directory(base_dir,
                                                          target_size=(300, 300),
                                                          batch_size=120,
                                                          class_mode='binary')
model = keras.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    layers.MaxPool2D(2, 2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPool2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])
history = model.fit_generator(train_generator,
                              steps_per_epoch=20,
                              epochs=5,
                              validation_data=validate_generator,
                              validation_steps=50, )
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epoch = range(len(acc))

plt.plot(epoch, acc, 'r', "training accuracy")
plt.plot(epoch, val_acc, 'b', "Validation accuracy")
plt.title("Training VS Valodation accuracy for number of epochs")
plt.figure()

plt.plot(epoch, loss, 'r', "training loss")
plt.plot(epoch, val_loss, 'b', "Validation loss")
plt.title('training and validation loss')

plt.show()
