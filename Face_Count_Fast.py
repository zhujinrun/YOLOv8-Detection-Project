import cv2
from ultralytics import YOLO
import csv
import os

# 实例化YOLO模型
model = YOLO('model/yolov8l-face.pt')

# 打开视频文件
video_path = "media/input.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频总帧数和FPS
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 设置处理的时间范围（以秒为单位，可以根据需要修改）
start_time = 0      # 开始时间（秒）
end_time = total_frames / fps  # 自动计算视频总时长（秒）

# 计算对应的帧范围
start_frame = start_time * fps
end_frame = min(end_time * fps, total_frames)  # 确保不超过总帧数

print(f"视频总帧数: {total_frames}")
print(f"FPS: {fps}")
print(f"处理帧范围: {start_frame} - {end_frame}")


# 计算每隔 n 秒的帧数间隔
frame_interval = fps * 15  # n 秒间隔
frame_count = 0

# 循环处理视频帧
while cap.isOpened():
    # 从视频中读取帧
    success, frame = cap.read()

    if success:
        frame_count += 1
        
        # 检查是否在指定的处理范围内
        if start_frame <= frame_count <= end_frame:
            # 每隔n秒处理一帧
            if frame_count % frame_interval == 1:
                # 使用YOLOv8进行人脸检测
                results = model(frame, conf=0.3)
                
                # 获取检测到的人脸数量
                face_count = len(results[0].boxes) if len(results) > 0 else 0
                
                # 计算当前时间戳（秒）
                timestamp = frame_count / fps
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)

                # 记录时间节点和抬头数量到CSV文件
                csv_file = 'result/students_count_fast.csv'
                file_exists = os.path.isfile(csv_file) and os.path.getsize(csv_file) > 0
                with open(csv_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists:
                        writer.writerow(['time_stamp', 'head_count'])
                    writer.writerow([f'{minutes:02d}:{seconds:02d}', face_count])
    else:
        # 视频结束
        break

# 释放资源
cap.release()
