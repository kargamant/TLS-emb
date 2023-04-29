import keras
import pandas as pd
import numpy as np
import sklearn
from keras import backend as K
import tensorflow as tf
from keras.layers import Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import pyarrow.parquet as pq
from numpy import array
import os
# from IPhyton.display import display
import copy
import math
from gensim.models import Word2Vec

table = pd.read_parquet('../parquets/train.parquet')
x_train, x_test, y_train, y_test=sklearn.model_selection.train_test_split(table, table['label'], test_size = 0.3, random_state=1)

culist = x_train['curves'].astype(str).tolist()
culist = [elem.replace('\'', '') for elem in culist]
culist = [elem.replace('[', '') for elem in culist]
culist = [elem.replace(']', '') for elem in culist]
culist = [elem.split() for elem in culist]

for i in range(len(culist)):
    for j in range(len(culist[i])):
        culist[i][j]='curve:'+culist[i][j]
#print(culist[0])

clist = x_train['ciphers'].astype(str).tolist()
# clist = [elem.replace('-', '') for elem in clist]
clist = [elem.replace('\'', '') for elem in clist]
clist = [elem.replace('[', '') for elem in clist]
clist = [elem.replace(']', '') for elem in clist]
clist = [elem.split() for elem in clist]

for i in range(len(clist)):
    clist[i]=clist[i]+culist[i]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(clist)
word_index = tokenizer.word_index
total_unique_words = len(tokenizer.word_index) + 1
#print(total_unique_words, math.sqrt(math.sqrt(total_unique_words)))
vocab = list(tokenizer.word_index.keys())
print(list(tokenizer.word_index.keys())[0])
mlen = 0
for i in clist:
    mlen = max(mlen, len(i))
for j in range(len(clist)):
    if len(clist[j]) < mlen:
        dif = mlen - len(clist[j])
        for _ in range(dif):
            clist[j].append('')
#print(clist[0])
data = tf.constant(clist)

strtovec = tf.keras.layers.StringLookup(max_tokens=total_unique_words, vocabulary=vocab)(data)
embedding = tf.keras.layers.Embedding(input_dim=total_unique_words, output_dim=4)(strtovec) #output dim 4
lstm = LSTM(4)(embedding)

model=Sequential()
#model.add(lstm)
x=Dense(64, activation='relu')(lstm)
x=Dense(64, activation='relu')(x)
x=Dense(64, activation='relu')(x)
#main_input = tf.keras.layers.Input(shape=(len(clist), len(clist[0])))
predictions = Dense(1, activation='sigmoid')(x)
model.add(predictions)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32)

score = model.evaluate(x_test, y_test, batch_size=64)
print(score)

#to be tested on similarity of vectors
# vec1=lstm[15]
# mdistance=math.inf
# mvec=0
# for i in range(len(lstm)):
#     vec2=lstm[i]
#     if i==15:
#         continue
#     dist=math.sqrt((vec2[0]-vec1[0])**2 + (vec2[1]-vec1[1])**2 + (vec2[2]-vec1[2])**2)
#     if dist<mdistance:
#         mdistance=dist
#         mvec=i
# print(clist[15])
# print(mvec, clist[mvec])
# print(embedding[:3])
# print(lstm[:3])

# auxiliary_output = Dense(1, activation='sigmoid')(lstm)
# print(auxiliary_output)





# print(total_unique_words)
# print(word_index)
# input_sequences = []
# for line in clist:
#     token_list = tokenizer.texts_to_sequences([line])[0]
#     for i in range(1, len(token_list)):
#         n_gram_seqs = token_list[:i + 1]
#         input_sequences.append(n_gram_seqs)
# print(len(input_sequences))
# print(input_sequences)
# max_seq_length = max([len(x) for x in input_sequences])
# input_seqs = np.array(pad_sequences(input_sequences, maxlen=max_seq_length, padding='pre'))
# print(max_seq_length)
# print(input_seqs[:5])
# x_values, labels = input_seqs[:, :-1], input_seqs[:, -1]
# y_values = tf.keras.utils.to_categorical(labels, num_classes=total_unique_words)
# print(x_values[56])
# print(labels[:3])
#



# model = Sequential()
#
# ciphers = tf.keras.Input(shape=(total_unique_words, ), name='ciphers')
# embedding = Embedding(output_dim=32, input_dim=total_unique_words)(ciphers)
# embed_lstm = LSTM(3)(embedding)
# embed_out = Dense(3, activation='sigmoid', name='embed_out')(embed_lstm)
#
#
# isbot = tf.keras.Input(shape=(1,), name='isbot')
# res = keras.layers.concatenate([isbot, embed_out])
#
# x = Dense(16, activation='relu')(res)
# x = Dense(16, activation='relu')(x)
# x = Dense(16, activation='relu')(x)
#
# main_output = Dense(1, activation='sigmoid', name='main_output')(x)
#
# model = tf.keras.Model(inputs=[ciphers, isbot], outputs=[embed_out, main_output])
#
# model.compile(optimizer='rmsprop',
#
#               loss={'main_output': 'binary_crossentropy', 'embed_out': 'binary_crossentropy'},
#
#               loss_weights={'main_output': 1., 'embed_out': 0.2})
#
# output = ()
# emb_output = ()
#
# # model.fit({'isbot': table['label'], 'ciphers': x_values},
# #
# #           {'main_output': output, 'embed_out': emb_output},
# #
# #           epochs=50, batch_size=32)
#
# pred = model.predict([table['label'], x_values])
#
# print(pred)