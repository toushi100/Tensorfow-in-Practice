import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as matimg
import os, zipfile
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image

local_zip = 'tmp/cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp')
zip_ref.close()

base_dir = '/tmp/cats_and_dogs_filtered'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory('/tmp/cats_and_dogs_filtered',
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')
validate_datagen = ImageDataGenerator(rescale=1. / 255)
validate_generator = validate_datagen.flow_from_directory('/tmp/cats_and_dogs_filtered',
                                                          target_size=(150, 150),
                                                          batch_size=20,
                                                          class_mode='binary')

model = keras.models.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
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
history = model.fit_generator(train_generator,
                                steps_per_epoch=100,
                                epochs=1,
                                validation_data=validate_generator,
                                validation_steps=50,
)

FN = 'sample_image.jpg'
path = 'content/' + FN
img = image.load_img(path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if classes[0] > 0.5:
    img = plt.imread(path)
    plt.imshow(img)
    plt.title('this is a cat')
    plt.show()
else:
    img = matimg.imread(path)
    plt.imshow(img)
    plt.title('this is a dog')
    plt.show()
