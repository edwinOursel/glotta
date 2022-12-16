import tensorflow_text as tf_text


class Preprocessor:
    def get_vocab(self, text):
        return self.get_vocab_chars(text)

    def get_vocab_chars(self, text):
        return sorted(set(text))

    def get_vocab_words(self, text):
        return tf_text.WhitespaceTokenizer().tokenize(text)
