import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

# 数据加载
def load_data(data_dir):
    images = []
    ages = []
    with open(os.path.join(data_dir, 'label.txt')) as f:
        for line in f:
            img_name, age = line.strip().split()
            img = tf.io.read_file(os.path.join(data_dir, img_name))
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [64, 64])
            images.append(img)
            ages.append(float(age))
    return tf.data.Dataset.from_tensor_slices((np.array(images), np.array(ages)))

# 构建模型
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
        layers.Dense(1, activation='relu')
    ])
    return model

# 训练流程
def train():
    dataset = load_data('data/imgdata').batch(32).prefetch(1)
    model = build_model()
    model.compile(optimizer='adam', loss='mse')
    
    for epoch in range(10):
        print(f'Epoch {epoch+1}')
        for step, (x_batch, y_batch) in enumerate(dataset):
            loss = model.train_on_batch(x_batch, y_batch)
            print(f'Step {step} loss: {loss:.4f}')
    
    model.save('age_prediction_model.h5')

if __name__ == '__main__':
    train()