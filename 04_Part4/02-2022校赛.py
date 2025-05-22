import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

# 数据加载和预处理
def load_data(data_dir):
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(data_dir, filename)
            img = tf.io.read_file(img_path)
            img = tf.image.decode_image(img, channels=3)
            img = tf.image.resize(img, [224, 224]) / 255.0
            
            # 根据文件名设置标签
            if filename.upper().startswith('P'):
                labels.append(1)  # 病理性近视
            else:
                labels.append(0)  # 非病理性近视
            images.append(img)
    return tf.data.Dataset.from_tensor_slices((np.array(images), np.array(labels)))

# 构建分类模型
def build_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(224,224,3)),
        layers.Conv2D(6, 5, strides=1),
        layers.ReLU(),
        layers.MaxPool2D(2,2),
        layers.Conv2D(16, 5, strides=1),
        layers.MaxPool2D(2,2),
        layers.ReLU(),
        layers.Conv2D(120, 4, strides=1, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')  # 二分类输出
    ])
    return model

# 训练流程
def train():
    dataset = load_data('data/2022campus').shuffle(1000).batch(32)
    model = build_model()
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    # 训练2个epoch以显示二次迭代结果
    history = model.fit(dataset, epochs=2, verbose=1)
    
    # 保存模型
    model.save('retinal_classification.h5')
    
    # 输出训练损失
    for epoch, loss in enumerate(history.history['loss']):
        print(f'Epoch {epoch+1} loss: {loss:.4f}')

if __name__ == '__main__':
    train()