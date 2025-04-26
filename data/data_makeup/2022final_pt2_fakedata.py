import numpy as np
import cv2
# import os
from PIL import Image, ImageDraw
import random

# 创建数据目录
# os.makedirs('data/2022final_pt2', exist_ok=True)

# ===== 任务1：TF-IDF文本数据 =====
task1_text = [
    "人工智能 大数据 云计算 人工智能 机器学习 深度学习 神经网络",
    "计算机视觉 自然语言处理 数据分析 大数据 人工智能 算法",
    "Python 编程 开发 测试 数据分析 人工智能 大数据",
    "深度学习 框架 神经网络 计算机视觉 自然语言处理 算法"
]
with open('data/2022final_pt2/2022final_pt2_fakedata_task1.txt', 'w') as f:
    f.write('\n'.join(task1_text))

# ===== 任务2：ORB特征图片数据 =====
# 生成400x400的测试图片
img1 = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.rectangle(img1, (50,50), (350,350), (0,255,0), -1)  # 绿色矩形

img2 = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.circle(img2, (200,200), 150, (0,0,255), -1)  # 红色圆形

cv2.imwrite('data/2022final_pt2/2022final_pt2_fakedata_task2_1.jpg', img1)
cv2.imwrite('data/2022final_pt2/2022final_pt2_fakedata_task2_2.jpg', img2)

# ===== 任务3：模板匹配数据 =====
# 生成模板（50x50红色圆形）
template = np.zeros((50, 50, 3), dtype=np.uint8)
cv2.circle(template, (25,25), 20, (0,0,255), -1)
cv2.imwrite('data/2022final_pt2/2022final_pt2_fakedata_task3_template.png', template)

# 生成目标图像（400x400随机位置嵌入模板）
target = np.random.randint(0, 255, (400,400,3), dtype=np.uint8)
x, y = random.randint(100,300), random.randint(100,300)
target[y:y+50, x:x+50] = template
cv2.imwrite('data/2022final_pt2/2022final_pt2_fakedata_task3_image.png', target)

print("所有伪造数据已生成于data/2022final_pt2目录")