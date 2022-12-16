import tensorflow_text as tf_text
from tensorflow.keras.layers.experimental import preprocessing
import re
import string
import tensorflow as tf

from tensorflow.keras import layers

text = "What you know you can't explain, but you feel it."

tokenizer = tf_text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(text)
print(tokens) # tensor

ids_from_words = preprocessing.StringLookup(vocabulary=tokens)

print(ids_from_words)

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)


train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label









