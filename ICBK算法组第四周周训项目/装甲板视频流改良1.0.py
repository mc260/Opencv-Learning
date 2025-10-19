import numpy as np
import cv2
from PIL import Image
from HSVfinding import get_limits

video=cv2.VideoCapture('text.mp4')

#red
#fingcolar=[33,52,167]
#blue
fingcolar=[247,245,66]

# 定义目标宽高比和允许的误差范围 - 新增参数
target_aspect_ratio = 0.37  # 例如：1.5表示宽高比为3:2
aspect_ratio_tolerance = 0.2  # 允许的宽高比误差范围


'''
hsv_img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lowerLimit,upperLimit = get_limits(color=fingcolar)
mask=cv2.inRange(hsv_img,lowerLimit,upperLimit)
mask_pil=Image.fromarray(mask)
bbox=mask_pil.getbbox()
print(bbox)
if bbox is not None:
     x1,y1,x2,y2=bbox
     cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),10)
'''
ret = True
while ret:
    ret, frame = video.read()
    hsv_video=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lowerLimit, upperLimit = get_limits(color=fingcolar)

    mask = cv2.inRange(hsv_video, lowerLimit, upperLimit)
    mask_pil = Image.fromarray(mask)

    # 形态学操作去除噪声和小物体
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算去噪声
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算填充空洞

    # 查找轮廓 - 修改部分开始
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 设置最小面积阈值，避免检测到噪声
    min_area = 0

    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)

        # 如果面积大于阈值，则绘制边界框
        if area > min_area:
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 计算当前边界框的宽高比 - 新增计算
            aspect_ratio = w / h

            # 计算宽高比与目标宽高比的差异 - 新增计算
            aspect_ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        # 检查宽高比是否在允许的误差范围内 - 新增条件
        if aspect_ratio_diff <= aspect_ratio_tolerance:
            # 在原始帧上绘制矩形
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)

    bbox = mask_pil.getbbox()
    reto, one = video.read()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(one, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.imshow('one', one)  # 播放每一帧

    cv2.imshow('frame', frame)           #播放每一帧
    cv2.imshow('mask', mask)
    if cv2.waitKey(10) & 0xFF == ord('q'):        #若触发键盘Q建 退出视频
        break

video.release()
cv2.destroyAllWindows()