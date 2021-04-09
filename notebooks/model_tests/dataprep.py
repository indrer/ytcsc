import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from gensim.models import Word2Vec, KeyedVectors



seed = 42
_word_vectors_file = '../../models/w2v/word2vec.wordvectors'
_word_vectors_file_stem = '../../models/w2v/word2vec_stem.wordvectors'
_word_vectors_file_long = '../../models/w2v/word2vec_long.wordvectors'

def get_vectors(wv_type=0):
    if wv_type == 0:
        return KeyedVectors.load(_word_vectors_file, mmap='r')
    elif wv_type == 1:
        return KeyedVectors.load(_word_vectors_file_stem, mmap='r')
    elif wv_type == 2:
        return KeyedVectors.load(_word_vectors_file_long, mmap='r')
    else:
        print('0 - word vectors for not stemmed data,\n1 - word vectors for stemmed data,\n2 - word vectors for longer comments')


def get_data(cmt_pos, cmt_neg, wv_type):
    # Drop random negative indices
    negative_indices = cmt_neg.index.tolist()
    diff = abs(len(cmt_neg) - len(cmt_pos))
    indices = np.random.choice(negative_indices, diff, replace=False)
    cmt_neg = cmt_neg.drop(indices)
    df = pd.concat([cmt_pos, cmt_neg], ignore_index=True)
    
    # Get the length of vectors
    vectors = get_vectors(wv_type)
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





