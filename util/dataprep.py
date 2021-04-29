import pandas as pd
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from gensim.models import Word2Vec, KeyedVectors



seed = 42
_word_vectors_file = '../../models/w2v/word2vec.wordvectors'
_word_vectors_file_stem = '../../models/w2v/word2vec_stem.wordvectors'
_word_vectors_file_long = '../../models/w2v/word2vec_long.wordvectors'
_glove_vectors = '../../glove.twitter.27B.200d.txt'

def get_glove_embeddings(word_index):
    glove_vectors = open(os.path.join(_glove_vectors), encoding="utf8")
    embeddings_index = {}
    for line in glove_vectors:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    glove_vectors.close()
    embedding_matrix = np.zeros((len(word_index) + 1, 200))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def get_vectors(wv_type=0, word_index=None):
    if wv_type == 0:
        return KeyedVectors.load(_word_vectors_file, mmap='r')
    elif wv_type == 1:
        return KeyedVectors.load(_word_vectors_file_stem, mmap='r')
    elif wv_type == 2:
        return KeyedVectors.load(_word_vectors_file_long, mmap='r')
    elif wv_type == 3:
        return get_glove_embeddings(word_index)
    else:
        print('0 - word vectors for not stemmed data,\n1 - word vectors for stemmed data,\n2 - word vectors for longer comments')


def get_data(cmt_pos, cmt_neg, wv_type):
    # Drop random negative indices
    negative_indices = cmt_neg.index.tolist()
    diff = abs(len(cmt_neg) - len(cmt_pos))
    indices = np.random.choice(negative_indices, diff, replace=False)
    cmt_neg = cmt_neg.drop(indices)
    df = pd.concat([cmt_pos, cmt_neg], ignore_index=True)
    
    sentences = [[word for word in str(body).split()] for body in df.body]

    #Set up tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index

    # Get the length of vectors
    vectors = get_vectors(wv_type, word_index)
    if wv_type == 3:
        vocab_size = len(word_index) + 1
        emdedding_size = 200
    else:
        vocab_size = len(vectors.vocab)
        emdedding_size = vectors.vector_size
        vectors = vectors.vectors

    X = tokenizer.texts_to_sequences(sentences)
    X = sequence.pad_sequences(X)
    Y= df.rating.values
    return X, Y, vocab_size, emdedding_size, vectors





