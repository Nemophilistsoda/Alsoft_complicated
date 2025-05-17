# 进行Harris角点检测并显示
images = []
for filename in os.listdir('user/Q2/2'):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join('user/Q2/2', filename)
        img = Image.open(img_path).convert('RGB')
        
        # 转换为NumPy数组进行Harris角点检测
        img_np = np.array(img)
        gray = np.mean(img_np, axis=2).astype(np.float32)
        
        # 计算Harris角点
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        Ixx = sobelx ** 2
        Iyy = sobely ** 2
        Ixy = sobelx * sobely
        
        # 使用高斯函数进行平滑
        Ixx = cv2.GaussianBlur(Ixx, (5, 5), 0)
        Iyy = cv2.GaussianBlur(Iyy, (5, 5), 0)
        Ixy = cv2.GaussianBlur(Ixy, (5, 5), 0)
        
        # 计算Harris响应
        k = 0.04
        det = Ixx * Iyy - Ixy ** 2
        trace = Ixx + Iyy
        harris_response = det - k * (trace ** 2)
        
        # 寻找角点
        corner_threshold = 0.01 * harris_response.max()
        corner_points = np.argwhere(harris_response > corner_threshold)
        
        # 在图像上绘制角点
        draw = ImageDraw.Draw(img)
        for point in corner_points:
            y, x = point
            draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(255, 0, 0))
        
        images.append(img)