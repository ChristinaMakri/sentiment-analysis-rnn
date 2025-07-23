from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_tokenizer(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer

def pad_sequences_fixed(sequences, maxlen=100):
    return pad_sequences(sequences, maxlen=maxlen, padding='post')
