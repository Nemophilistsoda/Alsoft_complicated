import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler

# ==================== 数据加载 ====================
data_path = 'data/task3data.csv'
data = pd.read_csv(data_path, header=None)
X = data.iloc[:, :30].values  # 前30列为特征
y = data.iloc[:, 30].values   # 第31列为标签

# ==================== 数据预处理 ====================
# 最大值最小值归一化 (0-1范围)
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# ==================== 数据集划分 ====================
X_train, X_test = X_normalized[:500], X_normalized[500:]
y_train, y_test = y[:500], y[500:]

# ==================== 模型训练 ====================
model = svm.SVC(kernel='linear')  # 使用线性核SVM
model.fit(X_train, y_train)

# ==================== 模型评估 ====================
accuracy = model.score(X_test, y_test)
print(f'测试集分类准确率: {accuracy:.4f}')

# 可选：输出更详细的分类报告
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print('\n分类报告:')
print(classification_report(y_test, y_pred))