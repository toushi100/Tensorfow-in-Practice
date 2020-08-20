import tensorflow as tf
import mitdeeplearning as mdl
import cv2
import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm

songs = mdl.lab1.load_training_data()

example_song = songs[0]
print("\nExample song:")
print(example_song)
mdl.lab1.play_song(example_song)

songs_joined = "\n\n".join(songs)
vocab = sorted(set(songs_joined))
print("there are", len(vocab), "unique characters int the dataset")

char2idx = {u: i for i, u in enumerate(vocab)}

idx2char = np.array(vocab)

print('{')
for char, _ in zip(char2idx, idx2char):
    print('{:4s}:{:3d},'.format(repr(char), char2idx[char]))
print(' ...\n')


def vectorize_string(string):
    vectorized_output = np.array([char2idx[char] for char in string])
    return vectorized_output


vectorized_songs = vectorize_string(songs_joined)

print('{} ---- charachters mapped to int ---> {}'.format(repr(songs_joined[:10]), vectorized_songs[:10]))


def get_batch(vectorized_songs, seq_length, batch_size):
    n = vectorized_songs.shape[0] - 1
    idx = np.random.choice((n - seq_length, batch_size))
    input_batch = [vectorized_songs[i + 1: i + seq_length + 1] for i in idx]
    output_batch = [vectorized_songs[i + 1: i + seq_length + i] for i in idx]

    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
    return x_batch, y_batch


test_args = (vectorized_songs, 10, 2)
if not mdl.lab1.test_batch_func_types(get_batch, test_args) or \
        not mdl.lab1.test_batch_func_shapes(get_batch, test_args) or \
        not mdl.lab1.test_batch_func_next_step(get_batch, test_args):
    print("======\n[FAIL] could not pass tests")
else:
    print("======\n[PASS] passed all tests!")

x_batch, y_batch = get_batch(vectorized_songs, seq_length=5, batch_size=1)
for i, (input_idx, target_idx) in enumerate(zip(np.squeeze(x_batch), np.squeeze(y_batch))):
    print("Step {:3d}".format(i))
    print("   input : {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
    print("   expected output : {} ({:s})".format(target_idx, repr(idx2char[target_idx])))


def LSTM(rnn_units):
    return tf.keras.layers.LSTM(
        rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=True,
    )


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        LSTM(rnn_units),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=32)
model.summary()

x, y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
pred = model(x)
print("input shape:   ", x.shape, " # (batch_size, sequence_length)")
print("prediction ", pred.shape, "  # (batch_size, sequence_length, vocab_size)")

sampled_indices = tf.random.categorical(pred[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
sampled_indices

print("Input: \n", repr("".join(idx2char[x[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))


def compute_loss(labels, logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return loss

example_batch_loss = compute_loss(y, pred)

print("Prediction shape : ", pred.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss :      ", example_batch_loss.numpy().mean())

num_training_iterations = 2000
batch_size = 4
seq_length = 100
learning_rate = 5e-3
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)
optimizer = tf.keras.optimizers.Adam(learning_rate)


def train_step(x, y):
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = compute_loss(y, y_hat)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


history = []
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
if hasattr(tqdm, '_instances'): tqdm._instances.clear()

for iter in tqdm(range(num_training_iterations)):
    x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)
    loss = train_step(x_batch, y_batch)
    history.append(loss.numpy().mean())
    plotter.plot(history)
    if iter % 100 == 0:
        model.save_weights(checkpoint_prefix)

model.save_weights(checkpoint_prefix)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()


def generate_text(model, start_string, generation_length=1000):
    input_eval = [char2idx[s] for s in start_string]  # TODO
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()
    tqdm._instances.clear()

    for i in tqdm(range(generation_length)):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    return (start_string + ''.join(text_generated))


generated_text = generate_text(model, start_string="X", generation_length=1000)

generated_songs = mdl.lab1.extract_song_snippet(generated_text)

for i, song in enumerate(generated_songs):
    waveform = mdl.lab1.play_song(song)
    if waveform:
        print("Generated song", i)
        ipythondisplay.display(waveform)
