from kalmanfilter import KalmanFilter
import cv2
import numpy as np

# 初始化卡尔曼滤波器
kf = KalmanFilter()

# 读取背景图（增加容错：读取失败则创建白色背景）
img = np.ones((400, 1200, 3), dtype=np.uint8) * 255  # 白色背景（高400，宽1200）

# 小球位置序列
ball_positions1 = [(50, 100), (100, 100), (150, 100), (200, 100), (250, 100), (300, 100), (350, 100), (400, 100), (450, 100)]
ball_positions2 = [(4, 300), (61, 256), (116, 214), (170, 180), (225, 148), (279, 120), (332, 97),
                   (383, 80), (434, 66), (484, 55), (535, 49), (586, 49), (634, 50),
                   (683, 58), (731, 69), (778, 82), (824, 101), (870, 124), (917, 148),
                   (962, 169), (1006, 212), (1051, 249), (1093, 290)]

# 绘制实测位置 + 卡尔曼预测位置
predicted = (0, 0)  # 初始化预测坐标
for pt in ball_positions2:
    # 绘制实测小球
    cv2.circle(img, pt, 15, (0, 20, 220), -1)
    # 卡尔曼预测（返回整数元组）
    predicted = kf.predict(pt[0], pt[1])
    # 绘制预测位置（绿色空心）
    cv2.circle(img, predicted, 15, (20, 220, 0), 4)

# 继续预测后续10帧位置
for i in range(10):
    predicted = kf.predict(predicted[0], predicted[1])
    cv2.circle(img, predicted, 15, (20, 220, 0), 4)

print(predicted)

# 显示图像（调整窗口大小，避免图片过大）
cv2.namedWindow("Img", cv2.WINDOW_NORMAL)
cv2.imshow("Img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()