import tensorflow as tf


class CorpusReader:
    def get_text(self):
        #file = tf.keras.utils.get_file('shakespeare.txt',
        #                               'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

        # Read, then decode for py2 compat.
        #text = open(file, 'rb').read().decode(encoding='utf-8')
        f = open("shakespeare.txt", "rb")
        text = f.read().decode(encoding='utf-8')
        print(text[0:200])
        return text
