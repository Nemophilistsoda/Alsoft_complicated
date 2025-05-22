import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 

# 读取原始数据 
X = [] 
f = open('task3.txt') 
lineIndex = 1 
for v in f: 
    if lineIndex > 1: 
        X.append([float(v.split()[1]), float(v.split()[2])]) 
    lineIndex += 1 
f.close()  # 关闭文件

# 转化为numpy array 
X = np.array(X) 

# 类簇的数量 
n_clusters = 10 

# 需要选手补全部分
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 可视化聚类结果
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
for i in range(n_clusters):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i+1}')

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='w', zorder=10)
plt.title('China')
plt.legend()
plt.show()