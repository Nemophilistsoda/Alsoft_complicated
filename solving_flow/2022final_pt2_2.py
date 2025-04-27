import cv2
import numpy as np
import os


# 1. 读取图片
img1 = cv2.imread('data/2022final_pt2/2022final_pt2_fakedata_task2_1.jpg')
img2 = cv2.imread('data/2022final_pt2/2022final_pt2_fakedata_task2_2.jpg')

# 2. 初始化ORB检测器
orb = cv2.ORB_create(nfeatures=1000)

# 3. 检测关键点并计算描述符
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 4. 显示关键点
img_kp1 = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
img_kp2 = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)

cv2.imshow('ORB Features 1', img_kp1)
cv2.imshow('ORB Features 2', img_kp2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存特征显示结果
cv2.imwrite('result/2022final_pt2_2/orb_features_1.jpg', img_kp1)
cv2.imwrite('result/2022final_pt2_2/orb_features_2.jpg', img_kp2)

# 5. 特征匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 6. 图像拼接
# 提取匹配点坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

# 计算单应性矩阵
H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

# 执行透视变换拼接
height, width = img1.shape[:2]
result = cv2.warpPerspective(img2, H, (width*2, height))
result[0:height, 0:width] = img1

# 显示并保存结果
cv2.imshow('Stitched Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('result/2022final_pt2_2/stitched_result.jpg', result)