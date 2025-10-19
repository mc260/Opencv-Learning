import cv2
import numpy as np

# 创建一个800x600的白色背景图像
canvas = np.ones((600, 800, 3), dtype=np.uint8) * 255


# 1. 绘制矩形 - 红色边框，绿色填充
cv2.rectangle(canvas, (50, 50), (200, 200), (0, 0, 255), 2)  # 红色边框矩形
cv2.rectangle(canvas, (60, 60), (190, 190), (0, 255, 0), -1)  # 绿色填充矩形

# 2. 绘制圆形 - 蓝色边框，黄色填充
cv2.circle(canvas, (300, 125), 50, (255, 0, 0), 3)  # 蓝色边框圆形
cv2.circle(canvas, (300, 125), 40, (0, 255, 255), -1)  # 黄色填充圆形

# 3. 绘制椭圆 - 紫色边框，青色填充
cv2.ellipse(canvas, (500, 125), (80, 40), 45, 0, 360, (255, 0, 255), 2)  # 紫色边框椭圆
cv2.ellipse(canvas, (500, 125), (70, 30), 45, 0, 360, (255, 255, 0), -1)  # 青色填充椭圆

# 4. 绘制直线 - 橙色
cv2.line(canvas, (50, 250), (750, 250), (0, 165, 255), 3)

# 5. 绘制箭头
cv2.arrowedLine(canvas, (300, 400), (450, 400), (0, 0, 0), 3, tipLength=0.1)

# ==================== 添加文字 ====================

# 设置字体
font_scale = 1.5
font_color = (0, 0, 0)  # 黑色
font_thickness = 3

# 1. 简单文字
cv2.putText(canvas, 'Hello ICBK', (300, 200),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)

# 2. 带样式的文字 - 使用不同字体
text = 'Hello ICBK'
text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1.8, 3)[0]

# 在文字下方绘制背景矩形
text_x, text_y = 250, 350
cv2.rectangle(canvas,
             (text_x - 10, text_y - text_size[1] - 10),
             (text_x + text_size[0] + 10, text_y + 10),
             (200, 200, 200), -1)  # 灰色背景
cv2.rectangle(canvas,
             (text_x - 10, text_y - text_size[1] - 10),
             (text_x + text_size[0] + 10, text_y + 10),
             (0, 0, 0), 2)  # 黑色边框

# 绘制带背景的文字
cv2.putText(canvas, text, (text_x, text_y),
            cv2.FONT_HERSHEY_COMPLEX, 1.8, (255, 0, 0), font_thickness)

# 3. 旋转文字 - 使用不同的字体样式
text_rotated = 'Hello ICBK'
font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
matrix = cv2.getRotationMatrix2D((550, 450), 30, 1)  # 旋转30度
cv2.putText(canvas, text_rotated, (550, 450), font, 1.5, (0, 100, 0), 2)

# 4. 带阴影效果的文字
text_shadow = 'Hello ICBK'
# 绘制阴影
cv2.putText(canvas, text_shadow, (302, 502),
            cv2.FONT_HERSHEY_DUPLEX, 2, (100, 100, 100), 3)
# 绘制前景文字
cv2.putText(canvas, text_shadow, (300, 500),
            cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)

# ==================== 显示和保存结果 ====================

# 显示图像
cv2.imshow('OpenCV Drawing - Hello ICBK', canvas)

# 等待按键
print("按任意键关闭窗口...")
cv2.waitKey(0)

# 保存图像
cv2.imwrite('hello_icbk_drawing.png', canvas)
print("图像已保存为 'hello_icbk_drawing.png'")

# 关闭所有窗口
cv2.destroyAllWindows()