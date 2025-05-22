import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd

# 数据加载
train_df = pd.read_csv('data/task4/trainlabels.txt', header=None)
test_df = pd.read_csv('data/task4/testlabels.txt', header=None)

# 图片预处理函数
def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [64, 64])
    return tf.cast(img, tf.float32) / 255.0

# 构建数据管道
train_paths = train_df.iloc[:,0].values
train_labels = train_df.iloc[:,1:8].values.astype(np.float32)
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.shuffle(100).map(
    lambda x,y: (load_and_preprocess_image(x), y))

# 构建模型
model = tf.keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='sigmoid')
])

# 编译模型
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, weight_decay=0.0001)
model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['accuracy'])

# 训练模型
model.fit(train_dataset.batch(32),
          epochs=10,
          verbose=1)

# 测试集评估
test_paths = test_df.iloc[:,0].values
test_labels = test_df.iloc[:,1:8].values.astype(np.float32)
test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_dataset = test_dataset.map(
    lambda x,y: (load_and_preprocess_image(x), y)).batch(32)

loss, acc = model.evaluate(test_dataset)
print(f'Test loss: {loss}, Test accuracy: {acc}')