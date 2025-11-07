import cv2
from ultralytics import YOLO

# 实例化YOLO模型
model = YOLO('model/yolov8l-face.pt')

# 打开视频文件
video_path = "media/input.mp4"
cap = cv2.VideoCapture(video_path)

# 添加视频写入器
fourcc = cv2.VideoWriter_fourcc(*'avc1')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('result/students_count.mp4', fourcc, fps, (width, height))

frame_count = 0

# 循环处理视频帧
while cap.isOpened():
    # 从视频中读取帧
    success, frame = cap.read()

    if success:
        frame_count += 1
        
        # 使用YOLOv8进行人脸检测
        results = model(frame, conf=0.3)
        
        # 获取检测到的人脸数量
        face_count = len(results[0].boxes) if len(results) > 0 else 0
        
        # 在图像上绘制检测框和人数统计
        if len(results) > 0 and hasattr(results[0].boxes, 'xyxy'):
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                # 绘制人脸边界框
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 计算当前时间戳（秒）
        timestamp = frame_count / fps
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        
        # 显示人脸数量和时间戳
        cv2.putText(frame, f'Time: {minutes:02d}:{seconds:02d} Raise Head Count: {face_count}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 写入输出视频
        out.write(frame)
        
        # 显示当前帧
        cv2.imshow("Statistics of student head counts", frame)
        
        # 按ESC退出
        if cv2.waitKey(30) & 0xFF == 27:
            break
    else:
        # 视频结束
        break

# 释放资源
cap.release()
# 释放视频写入器
out.release()
cv2.destroyAllWindows()
