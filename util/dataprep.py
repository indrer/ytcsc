import pandas as pd
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from gensim.models import Word2Vec, KeyedVectors



seed = 42
_glove_vectors = '../../glove.twitter.27B.200d.txt'
_train_path = '../../datasets/split/train.csv'
_test_path = '../../datasets/split/test.csv'
_val_path ='../../datasets/split/val.csv'

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

def get_vectors(word_index=None):
    return get_glove_embeddings(word_index)

def get_data():
    train = pd.read_csv(_train_path)
    test = pd.read_csv(_test_path)
    val = pd.read_csv(_val_path)
    df = pd.concat([train, test, val], ignore_index=True)
    sentences = [[word for word in str(body).split()] for body in df.body]

    #Set up tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index

    # Get the length of vectors
    vectors = get_vectors(word_index)
    vocab_size = len(word_index) + 1
    emdedding_size = 200

    # Make data into tokens
    X = tokenizer.texts_to_sequences(sentences)
    X = sequence.pad_sequences(X)

    # Prepare train data
    X_train = X[0:len(train)]
    y_train = train.rating.values

    # Prepare test data
    test_index = (len(train)) + (len(test))
    X_test = X[len(train):test_index]
    y_test = test.rating.values

    # Prepare val data
    X_val = X[test_index:]
    y_val = val.rating.values

    return X_train, y_train, X_test, y_test, X_val, y_val, vocab_size, emdedding_size, vectors





