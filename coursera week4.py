import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, zipfile, requests
import urllib.request
import tkinter as tk
from tkinter import filedialog

print("begining download")
url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip'
local_zip = '/tmp/horse-or-human.zip'
urllib.request.urlretrieve(url, local_zip)
print("download done")
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/horse-or-human')
zip_ref.close()

train_horse_dir = os.path.join('/tmp/horse-or-human/horses')
train_human_dir = os.path.join('/tmp/horse-or-human/humans')

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])
train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

print('total training horse images:', len(os.listdir(train_horse_dir)))
print('total training human images:', len(os.listdir(train_human_dir)))

train_datagen = ImageDataGenerator(rescale=1. / 255)
train_dir = '/tmp/horse-or-human/'
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(300, 300), batch_size=128, class_mode='binary'
)

model = tf.keras.models.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation='relu',
                        input_shape=(300, 300, 3)),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=3,
    validation_steps=8,
    verbose=2
)
model.save('horse-or-human.h5')

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()

path = file_path
img = image.load_img(path, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if classes[0] > 0.5:
    print(path + "is a human")
else:
    print(path + "is a horse")
