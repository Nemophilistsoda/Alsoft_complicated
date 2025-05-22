import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import cv2

# 数据增强生成函数
def generate_rotated_images(images_dir):
    images = []
    angles = []
    for img_name in os.listdir(images_dir):
        img_path = os.path.join(images_dir, img_name)
        img = cv2.imread(img_path)
        
        # 生成10个随机旋转样本
        for _ in range(10):
            angle = np.random.uniform(-60, 60)
            rows, cols = img.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
            rotated = cv2.warpAffine(img, M, (cols, rows))
            
            images.append(cv2.resize(rotated, (64,64)) / 255.0)
            angles.append(angle)
    return tf.data.Dataset.from_tensor_slices((np.array(images), np.array(angles)))

# 构建回归模型
def build_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(64,64,3)),
        layers.Conv2D(8, 3, activation='relu'),
        layers.MaxPool2D(2),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPool2D(2),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPool2D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1)  # 回归输出角度值
    ])
    return model

# 训练流程
def train():
    dataset = generate_rotated_images('data/task4').batch(32).shuffle(1000)
    
    # 拆分训练测试集
    train_size = int(0.8 * len(dataset))
    train_data = dataset.take(train_size)
    test_data = dataset.skip(train_size)
    
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