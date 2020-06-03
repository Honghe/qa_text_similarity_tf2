# -*- coding: utf-8 -*-
import datetime
import itertools
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow import keras

# 可复现性
np.random.seed(0)
tf.random.set_seed(0)

# 加载数据
df = pd.read_csv('./data/atec_nlp.csv', sep='\t', names=['question1', 'question2', 'label'])

# 分割成训练集与测试集
train_df, test_df, y_train, y_test = train_test_split(pd.DataFrame(df, columns=['question1', 'question2']), df['label'],
                                                      test_size=0.2)

# 对数据集的汉字进行Embedding

# 加载Word2Vec，每个汉字128维的向量
df = pd.read_csv('./data/char_vec', sep='\s+', header=None, index_col=0)
# 行的index即汉字
alphabet = df.index.values  # len is 1575
vocabulary = dict([(alphabet[i], i) for i in range(len(alphabet))])
# 添加占位符
vocabulary['<u>'] = len(vocabulary)
inverse_vocabulary = alphabet + ['<u>']  # <u> 是占位符

embedding_dim = 128
embeddings = np.random.randn(len(vocabulary), embedding_dim)  # embedding matrix
# 填充
for i, (name, data) in enumerate(df.iterrows()):
    embeddings[i] = data.to_numpy()

# 汉字转成数字id，此步骤需要一些时间
for dataset in [train_df, test_df]:
    for index, row in dataset.iterrows():
        # 两列句子都处理
        for question in ['question1', 'question2']:
            question2n = []  # question2n -> question numbers representation
            # TODO 更好的切分汉字与其它字符
            for word in row[question]:
                if word not in vocabulary:
                    question2n.append(vocabulary['<u>'])
                else:
                    question2n.append(vocabulary[word])

            # 逐个将文本替换为数字
            dataset.at[index, question] = question2n

# 准备training与validation数据
max_seq_length = max(train_df['question1'].map(lambda x: len(x)).max(),
                     train_df['question2'].map(lambda x: len(x)).max(),
                     train_df['question1'].map(lambda x: len(x)).max(),
                     train_df['question2'].map(lambda x: len(x)).max())
print('max_seq_length {}'.format(max_seq_length))

# 分割为train与validation
validation_size = 10000
training_size = len(train_df) - validation_size
questions_cols = ['question1', 'question2']

X_train, X_validation, Y_train, Y_validation = train_test_split(train_df, y_train, test_size=validation_size)

# Split to dicts
X_train = {'left': X_train.question1, 'right': X_train.question2}
X_validation = {'left': X_validation.question1, 'right': X_validation.question2}
X_test = {'left': test_df.question1, 'right': test_df.question2}

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

# Zero padding
for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
    dataset[side] = tf.keras.preprocessing.sequence.pad_sequences(dataset[side], maxlen=max_seq_length)

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

# 构建模型
# Model variables
n_hidden = 30
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 5


def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))


# The visible layer
left_input = keras.Input(shape=(max_seq_length,), dtype='float32')
right_input = keras.Input(shape=(max_seq_length,), dtype='float32')

# 使用已定义好的embedding amtrix
embedding_layer = keras.layers.Embedding(len(embeddings), embedding_dim,
                                         embeddings_initializer=tf.keras.initializers.Constant(embeddings),
                                         input_length=max_seq_length, trainable=False)

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = keras.layers.LSTM(n_hidden)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = keras.layers.Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                                      output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# Pack it all up into a model
malstm = keras.models.Model([left_input, right_input], [malstm_distance])

# Adadelta optimizer, with gradient clipping by norm
optimizer = keras.optimizers.Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

# Start training
training_start_time = time()

malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, epochs=n_epoch,
                            validation_data=([X_validation['left'], X_validation['right']], Y_validation))

print("Training time finished.\n{} epochs in {}".format(n_epoch,
                                                        datetime.timedelta(seconds=time() - training_start_time)))

# 画出过程
# Plot accuracy
plt.plot(malstm_trained.history['accuracy'])
plt.plot(malstm_trained.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot loss
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
