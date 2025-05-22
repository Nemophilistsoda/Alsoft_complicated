import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os

# 多标签数据加载
def load_multilabel_data(data_dir, label_file):
    images = []
    labels = []
    with open(os.path.join(data_dir, label_file)) as f:
        for line in f:
            parts = line.strip().split()
            img_path = os.path.join(data_dir, parts[0])
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [64, 64]) / 255.0
            images.append(img)
            labels.append(np.array(parts[1:8], dtype=np.float32))
    return tf.data.Dataset.from_tensor_slices((np.array(images), np.array(labels)))

# 构建多标签分类模型
def build_model():
    inputs = layers.Input(shape=(64,64,3))
    x = layers.Conv2D(8, 3, strides=1, activation='relu')(inputs)
    x = layers.MaxPool2D(2,2)(x)
    x = layers.Conv2D(16, 3, strides=1)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(2,2)(x)
    x = layers.Conv2D(32, 3, strides=1, activation='relu')(x)
    x = layers.MaxPool2D(2,2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(7, activation='sigmoid')(x)  # 7个二分类输出
    return Model(inputs, outputs)

# 训练流程
def train():
    train_data = load_multilabel_data('data/task4', 'trainlabels.txt').batch(32)
    test_data = load_multilabel_data('data/task4', 'testlabels.txt').batch(32)
    
    model = build_model()
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',  # 多标签使用binary_crossentropy
                 metrics=['accuracy'])
    
    # 训练过程
    for epoch in range(10):
        print(f'Epoch {epoch+1}')
        for step, (x_batch, y_batch) in enumerate(train_data):
            loss = model.train_on_batch(x_batch, y_batch)
            print(f'Step {step} loss: {loss[0]:.4f}')
    
    # 测试评估
    test_loss, test_acc = model.evaluate(test_data)
    print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    train()