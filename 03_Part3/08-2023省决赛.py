import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

# 加载查询图像
query_img = cv2.imread('data/task3/image_0002.jpg')
sift = cv2.SIFT_create()
kp_query, des_query = sift.detectAndCompute(query_img, None)
des_query = des_query[:32] if len(des_query) > 32 else np.vstack([des_query, np.zeros((32-len(des_query),128))])

# 加载数据库图像
data_dir = 'data/task3/data'
image_files = [os.path.join(data_dir,f) for f in os.listdir(data_dir)]

# 存储相似度结果
similarities = []

for img_path in image_files:
    # 提取特征
    img = cv2.imread(img_path)
    kp, des = sift.detectAndCompute(img, None)
    if des is None:
        continue
    
    # 固定描述子数量为32
    des = des[:32] if len(des) > 32 else np.vstack([des, np.zeros((32-len(des),128))])
    
    # 计算余弦距离
    dist = cosine(des_query.flatten(), des.flatten())
    similarities.append((img_path, dist))

# 按相似度排序
similarities.sort(key=lambda x: x[1])
top3 = similarities[:3]

# 显示结果
plt.figure(figsize=(15,5))
plt.subplot(141), plt.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
plt.title('Query Image'), plt.axis('off')

for i, (path, dist) in enumerate(top3):
    img = cv2.imread(path)
    plt.subplot(142+i)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'Top {i+1} (dist={dist:.2f})'), plt.axis('off')

plt.tight_layout()
plt.show()