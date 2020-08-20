import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import IPython
import functools
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import mitdeeplearning as mdl

path_to_training_data = 'datasets/train_face.h5'

loader = mdl.lab2.TrainingDatasetLoader(path_to_training_data)

number_of_training_data = loader.get_train_size()
images, labels = loader.get_batch(100)

face_images = images[np.where(labels == 1)[0]]
not_face_images = images[np.where(labels == 0)[0]]

idx_face = 23
idx_not_face = 9

plt.figure(figsize=(5, 5))
plt.subplot(1, 2, 1)
plt.imshow(face_images[idx_face])
plt.title("face")
plt.grid(False)

plt.subplot(1, 2, 2)
plt.imshow(not_face_images[idx_not_face])
plt.title("NOT A FACE like .. .  clearly")
plt.grid(False)
plt.show()

n_filters = 12


def make_standard_classifier(n_outputs=1):
    Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='same', activation='relu')
    BatchNormalization = tf.keras.layers.BatchNormalization
    Flatten = tf.keras.layers.Flatten
    Dense = functools.partial(tf.keras.layers.Dense, activation='relu')

    model = tf.keras.Sequential([
        Conv2D(filters=1 * n_filters, kernel_size=5, strides=2),
        BatchNormalization(),

        Conv2D(filters=2 * n_filters, kernel_size=5, strides=2),
        BatchNormalization(),

        Conv2D(filters=4 * n_filters, kernel_size=3, strides=2),
        BatchNormalization(),

        Conv2D(filters=6 * n_filters, kernel_size=3, strides=2),
        BatchNormalization(),

        Flatten(),
        Dense(512),
        Dense(n_outputs, activation=None),
    ])
    return model


standard_classifier = make_standard_classifier()
batch_size = 32
num_epochs = 2  # keep small to run faster
learning_rate = 5e-4

optimizer = tf.keras.optimizers.Adam(learning_rate)  # define our optimizer
loss_history = mdl.util.LossHistory(smoothing_factor=0.99)  # to record loss evolution
plotter = mdl.util.PeriodicPlotter(sec=2, scale='semilogy')
if hasattr(tqdm, '_instances'): tqdm._instances.clear()  # clear if it exists


@tf.function
def standard_train_step(x, y):
    with tf.GradientTape() as tape:
        # feed the images into the model
        logits = standard_classifier(x)
        # Compute the loss
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

    # Backpropagation
    grads = tape.gradient(loss, standard_classifier.trainable_variables)
    optimizer.apply_gradients(zip(grads, standard_classifier.trainable_variables))
    return loss


# The training loop!
for epoch in range(num_epochs):
    for idx in tqdm(range(loader.get_train_size() // batch_size)):
        x, y = loader.get_batch(batch_size)
        loss = standard_train_step(x, y)
        loss_history.append(loss.numpy().mean())
        plotter.plot(loss_history.get())
(batch_x, batch_y) = loader.get_batch(5000)
y_pred_standard = tf.round(tf.nn.sigmoid(standard_classifier.predict(batch_x)))
acc_standard = tf.reduce_mean(tf.cast(tf.equal(batch_y, y_pred_standard), tf.float32))

print("Standard CNN accuracy on (potentially biased) training set: {:.4f}".format(acc_standard.numpy()))
test_faces = mdl.lab2.get_test_faces()
keys = ["Light Female", "Light Male", "Dark Female", "Dark Male"]
for group, key in zip(test_faces, keys):
    plt.figure(figsize=(5, 5))
    plt.imshow(np.hstack(group))
    plt.title(key, fontsize=15)

standard_classifier_logits = [standard_classifier(np.array(x, dtype=np.float32)) for x in test_faces]
standard_classifier_probs = tf.squeeze(tf.sigmoid(standard_classifier_logits))

xx = range(len(keys))
yy = standard_classifier_probs.numpy().mean(1)
plt.bar(xx, yy)
plt.xticks(xx, keys)
plt.ylim(max(0, yy.min() - yy.ptp() / 2.), yy.max() + yy.ptp() / 2.)
plt.title("Standard classifier predictions")
