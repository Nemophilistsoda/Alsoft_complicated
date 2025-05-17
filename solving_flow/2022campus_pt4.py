import tensorflow as tf
from tensorflow.keras import layers, Model
import os
import numpy as np

# ==================== 数据准备 ====================
def load_dataset(data_dir):
    """
    加载眼底图像数据集
    预处理流程：调整尺寸 → 归一化 → 构建数据管道
    """
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    labels = []
    for f in image_paths:
        if os.path.basename(f).startswith('P'):
            labels.append(1)
        else:
            labels.append(0)
    
    def preprocess_image(image_path, label):
        # 图像解码与预处理
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = (image / 127.5) - 1.0  # 归一化到[-1, 1]
        return image, label
    
    # 构建高效数据管道
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

# ==================== 模型构建 ====================
class RetinaModel(Model):
    """
    视网膜病变分类CNN模型
    网络结构：3个卷积块 → 2个全连接层 → Dropout正则化
    """
    def __init__(self):
        super().__init__()  # super() 用于调用父类的方法
        # 卷积块1：6个5x5卷积核 → 最大池化
        self.conv1 = layers.Conv2D(6, 5, padding='valid', activation='relu')
        self.pool1 = layers.MaxPool2D(2, 2)
        
        # 卷积块2：16个5x5卷积核 → 最大池化 
        self.conv2 = layers.Conv2D(16, 5, padding='valid', activation='relu')
        self.pool2 = layers.MaxPool2D(2, 2)
        
        # 卷积块3：120个4x4卷积核
        self.conv3 = layers.Conv2D(120, 4, padding='valid', activation='relu')
        
        # 全连接层
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(64, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.fc2 = layers.Dense(2, activation='relu')

    def call(self, inputs, training=False):
        # 前向传播过程
        x = self.conv1(inputs)    # 输出形状: (220, 220, 6)
        x = self.pool1(x)          # 输出形状: (110, 110, 6)
        x = self.conv2(x)          # 输出形状: (106, 106, 16)
        x = self.pool2(x)          # 输出形状: (53, 53, 16)
        x = self.conv3(x)          # 输出形状: (50, 50, 120)
        x = self.flatten(x)        # 展平为300000特征
        x = self.fc1(x)            # 全连接至64维
        x = self.dropout(x, training=training)
        return self.fc2(x)         # 输出2维特征

# ==================== 训练配置 ====================
# 初始化模型和优化器
model = RetinaModel()
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 加载数据
dataset = load_dataset('data/retina_images')

# ==================== 模型训练 ====================
print("开始模型训练...")
for epoch in range(2):
    epoch_loss = []
    for images, labels in dataset:
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        epoch_loss.append(loss.numpy())
    
    print(f'Epoch {epoch+1} 平均损失: {np.mean(epoch_loss):.4f}')