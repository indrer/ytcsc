import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from gensim.models import Word2Vec, KeyedVectors



seed = 42
_word_vectors_file = '../../models/w2v/word2vec.wordvectors'

def get_vectors():
    return KeyedVectors.load(_word_vectors_file, mmap='r')


def get_data(cmt_pos, cmt_neg):
    # Drop random negative indices
    negative_indices = cmt_neg.index.tolist()
    diff = abs(len(cmt_neg) - len(cmt_pos))
    indices = np.random.choice(negative_indices, diff, replace=False)
    cmt_neg = cmt_neg.drop(indices)
    df = pd.concat([cmt_pos, cmt_neg], ignore_index=True)
    
    # Get the length of vectors
    vectors = get_vectors()
    vocab_size = len(vectors.vocab)
    emdedding_size = vectors.vector_size
    del vectors

    sentences = [[word for word in str(body).split()] for body in df.body]

    #Set up tokenizer
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(sentences)

    X = tokenizer.texts_to_sequences(sentences)
    X = sequence.pad_sequences(X)
    Y= df.rating.values
    return X, Y, vocab_size, emdedding_size





