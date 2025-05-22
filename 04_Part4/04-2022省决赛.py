import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

# 加载关键点数据
def load_keypoints(data_dir, label_file):
    images = []
    keypoints = []
    with open(os.path.join(data_dir, label_file)) as f:
        for line in f:
            parts = line.strip().split()
            img_path = os.path.join(data_dir, parts[0])
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [64, 64]) / 255.0
            images.append(img)
            keypoints.append(np.array(parts[1:11], dtype=np.float32))
    return tf.data.Dataset.from_tensor_slices((np.array(images), np.array(keypoints)))

# 构建关键点回归模型
def build_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(64,64,3)),
        layers.Conv2D(8, 3, strides=1, activation='relu'),
        layers.MaxPool2D(2,2),
        layers.Conv2D(16, 3, strides=1),
        layers.ReLU(),
        layers.MaxPool2D(2,2),
        layers.Conv2D(32, 3, strides=1, activation='relu'),
        layers.MaxPool2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10)  # 10个坐标点输出
    ])
    return model

# 训练流程
def train():
    train_data = load_keypoints('data/task4', 'train.txt').batch(32)
    test_data = load_keypoints('data/task4', 'test.txt').batch(32)
    
    model = build_model()
    model.compile(optimizer='adam', loss='mae')
    
    # 训练过程
    for epoch in range(10):
        print(f'Epoch {epoch+1}')
        for step, (x_batch, y_batch) in enumerate(train_data):
            loss = model.train_on_batch(x_batch, y_batch)
            print(f'Step {step} loss: {loss:.4f}')
    
    # 测试评估
    test_loss = model.evaluate(test_data)
    print(f'Test MAE: {test_loss:.4f}')

if __name__ == '__main__':
    train()