# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据
df = pd.read_csv('./data/atec_nlp.csv', sep='\t', names=['q1', 'q2', 'label'])
# 分割成训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(df, columns=['q1', 'q2']), df['label'], test_size=0.2)

# 加载Word2Vec

