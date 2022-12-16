import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import time

from corpus_reader import CorpusReader
from fit import fit
from generator import Generator
from model import MyModel
from preprocessor import Preprocessor

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.test.gpu_device_name())
print(tf.__version__)

preprocessor = Preprocessor()
text = CorpusReader().get_text()
vocab = preprocessor.get_vocab(text)

ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab))  # strings into integer

#ids_from_words = preprocessing.TextVectorization().adapt(text.batch(64))


chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True)

all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

seq_length = 100
sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)


# TODO change this to get words as tokens ?
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = (
    dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
EMBEDDING_DIM = 256

# Number of RNN units
RNN_UNITS = 1024

model = MyModel(
    # Be sure the vocabulary size matches the `StringLookup` layers.
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=EMBEDDING_DIM,
    rnn_units=RNN_UNITS)

fit(model, dataset)


generator = Generator(model, chars_from_ids, ids_from_chars)

start = time.time()
states = None
next_char = tf.constant(['ROMEO:'])
result = [next_char]

for n in range(1000):
    next_char, states = generator.generate_one_step(next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()

print(result[0].numpy().decode('utf-8'), '\n\n' + '_' * 80)

print(f"\nRun time: {end - start}")
