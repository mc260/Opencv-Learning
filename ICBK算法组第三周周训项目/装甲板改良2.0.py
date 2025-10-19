import numpy as np
import cv2
from PIL import Image
from HSVfinding import get_limits

video=cv2.VideoCapture('text.mp4')

#red
#fingcolar=[33,52,167]
#blue
fingcolar=[247,245,66]

target_aspect_ratio =5  # 期望长宽比
tolerance = 3  # 容差范围

# 设置面积筛选条件
min_area = 100  # 最小面积阈值（像素）
max_area = 30000  # 最大面积阈值（像素）

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
    lowerLimit, upperLimit = get_limits(color=fingcolar) #获取HSV上下值

    mask = cv2.inRange(hsv_video, lowerLimit, upperLimit)

    # 形态学操作去除噪声和小物体
    kernel = np.ones((1,1), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算去噪声
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算填充空洞

    mask_pil = Image.fromarray(mask)                        #pil库获取矩形
    bbox = mask_pil.getbbox()                               #合成矩形
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        # 计算当前边界框的长宽比
        width = x2 - x1
        height = y2 - y1
        area = width * height  # 边界框面积
        # 避免除以零的错误
        if height > 0:
            current_aspect_ratio = width / height

            # 检查长宽比是否在目标范围内
            min_ratio = target_aspect_ratio - tolerance
            max_ratio = target_aspect_ratio + tolerance
            if (min_ratio <= current_aspect_ratio <= max_ratio and
                     min_area <= area <= max_area):
                    # 只有长宽比和面积符合要求的物体才会被绘制矩形
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 10)

    cv2.imshow('frame', frame)           #播放每一帧
    cv2.imshow('mask', mask)
    if cv2.waitKey(10) & 0xFF == ord('q'):        #若触发键盘Q建 退出视频
        break

video.release()
cv2.destroyAllWindows()