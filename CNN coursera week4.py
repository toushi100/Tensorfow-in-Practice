import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

base_dir = 'tmp/rps/'

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   rotation_range=40,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
training_generator = train_datagen.flow_from_directory(base_dir,
                                                       target_size=(150, 150),
                                                       batch_size=50,
                                                       class_mode='sparse')

validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_directory(base_dir,
                                                              target_size=(150, 150),
                                                              class_mode='sparse')
model = keras.Sequential([
    layers.Conv2D(16, (3, 3), input_shape=(150, 150, 3), activation='relu'),
    layers.MaxPool2D(2, 2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPool2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(3, activation='softmax')
])
model.summary()
model.compile(optimizer=RMSprop(lr=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

history = model.fit_generator(training_generator,
                              steps_per_epoch=100,
                              epochs=20,
                              validation_data=validation_generator,
                              validation_steps=50)
