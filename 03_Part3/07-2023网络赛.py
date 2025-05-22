import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import os

# 图像特征提取函数
def extract_features(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    _, des = sift.detectAndCompute(gray, None)
    return des if des is not None else np.array([])

# 加载数据集
data_dir = 'data/task3'
features = []
labels = []

for class_idx, class_name in enumerate(['cat', 'dog']):
    class_dir = os.path.join(data_dir, class_name)
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        des = extract_features(img_path)
        if des.any():
            features.extend(des)
            labels.extend([class_idx]*len(des))

# 构建视觉词袋
kmeans = KMeans(n_clusters=100)
kmeans.fit(features)

# 生成直方图特征
def img_to_histogram(img_path):
    des = extract_features(img_path)
    if des.any():
        visual_words = kmeans.predict(des)
        hist, _ = np.histogram(visual_words, bins=100, range=(0,99))
        return hist
    return np.zeros(100)

# 准备训练数据
X = []
y = []
for class_idx, class_name in enumerate(['cat', 'dog']):
    class_dir = os.path.join(data_dir, class_name)
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        hist = img_to_histogram(img_path)
        X.append(hist)
        y.append(class_idx)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# 训练SVM模型
svm = LinearSVC()
svm.fit(X_train, y_train)

# 评估模型
y_pred = svm.predict(X_test)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print(f'Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1-score: {f1:.2f}')