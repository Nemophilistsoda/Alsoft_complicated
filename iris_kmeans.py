import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 读取数据文件
data = np.loadtxt('data/task3.txt')
X = data[:, :3]  # 前三列作为特征
y = data[:, 3]   # 最后一列作为类别

# 任务1：原始数据可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for cls in np.unique(y):
    plt.scatter(X[y == cls, 0], X[y == cls, 1], label=f'Class {int(cls)}')
plt.title('Original Data')
plt.legend()

# 任务2：KMeans聚类
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_

# 任务3：聚类结果可视化
plt.subplot(1, 2, 2)
for i in range(3):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f'Cluster {i}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='x', s=200, linewidths=3, color='k', label='Centroids')
plt.title('KMeans Clustering Result')
plt.legend()
plt.tight_layout()
plt.show()