import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

sentences = [
    'I love my dog',
    'i love my cat',
    'you love my dog!',
    'do you think my dog is amazing'
]
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=8)

print(word_index)
print(sequences)
print(padded)

test_data = [
    'i really love my dog',
    'my dog loves my mantee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)
padded = pad_sequences(test_seq, maxlen=10)
print(padded)

with open("datasets/sarcasm/sarcasm.json", 'r') as f:
    datastore = json.load(f)


sentences = []
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sequences)
padded = pad_sequences(sequences, padded='post')
print(padded[0])
print(padded.shape)
