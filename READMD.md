# Text Similarity for Question Answer
Ref: https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb

# Dependencies
- TensorFlow 2.2

# Model
Siamese Network，use LSTM.

# Data Set

2018 蚂蚁金服NLP智能客服比赛，数据集大小200k条。

https://dc.cloud.alipay.com/index#/topic/intro?id=3

# Training Log

可以看出很早就拟合了，再训练增长很少。
```
Epoch 1/5
1925/1925 [==============================] - 19s 10ms/step - loss: 0.1479 - accuracy: 0.8165 - val_loss: 0.1507 - val_accuracy: 0.8114
Epoch 2/5
1925/1925 [==============================] - 19s 10ms/step - loss: 0.1478 - accuracy: 0.8166 - val_loss: 0.1506 - val_accuracy: 0.8115
Epoch 3/5
1925/1925 [==============================] - 19s 10ms/step - loss: 0.1477 - accuracy: 0.8168 - val_loss: 0.1504 - val_accuracy: 0.8115
```

