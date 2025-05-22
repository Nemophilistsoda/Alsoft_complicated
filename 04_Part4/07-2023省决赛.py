import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 加载数据
data = pd.read_csv('data/task4/task4.csv', header=None)
values = data[1].values.astype('float32')

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values.reshape(-1, 1))

# 创建滑动窗口数据集
def create_dataset(data, look_back=6):
    X, Y = [], []
    for i in range(len(data)-look_back):
        X.append(data[i:(i+look_back), 0])
        Y.append(data[i+look_back, 0])
    return np.array(X), np.array(Y)

X, y = create_dataset(scaled)

# 数据集划分
train_size = int(len(X) * 0.9)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# 调整输入形状 [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

# 构建LSTM模型
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(10, input_shape=(6, 1)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
model.compile(loss='mse', optimizer=optimizer)

# 训练模型
history = model.fit(X_train, y_train,
                   epochs=2,
                   validation_data=(X_val, y_val),
                   verbose=1,
                   batch_size=16)

# 输出训练过程
print('Training loss:', history.history['loss'])
print('Validation loss:', history.history['val_loss'])