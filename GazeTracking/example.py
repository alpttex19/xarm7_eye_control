"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import numpy as np
import os
import sys
import cv2
import pyrealsense2 as rs
from gaze_tracking import GazeTracking

# Set environment variables to avoid Qt threading issues
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

# Initialize CV2 with proper backend
cv2.setUseOptimized(True)
# 配置 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()

# 启用彩色和深度流
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# 开始流
pipeline.start(config)

# Try to set a specific backend to avoid Qt issues
try:
    cv2.setLogLevel(0)  # Suppress OpenCV warnings
except:
    pass

gaze = GazeTracking()

# Create window in main thread to avoid Qt warnings
cv2.namedWindow("Demo", cv2.WINDOW_NORMAL)

while True:
    # We get a new frame from the webcam
    # 等待一帧数据
    frames = pipeline.wait_for_frames()

    # 获取深度帧和彩色帧
    color_frame = frames.get_color_frame()

    if not color_frame:
        continue

    # 将帧转换为 numpy 数组
    frame = np.asanyarray(color_frame.get_data())
    original_height, original_width = frame.shape[:2]

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right(threshold=0.55):
        text = "Looking right"
    elif gaze.is_left(threshold=0.65):
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(
        frame,
        "Left pupil:  " + str(left_pupil),
        (90, 130),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (147, 58, 31),
        1,
    )
    cv2.putText(
        frame,
        "Right pupil: " + str(right_pupil),
        (90, 165),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (147, 58, 31),
        1,
    )
    cv2.putText(
        frame,
        "horizontal_ratio: " + str(gaze.horizontal_ratio()),
        (90, 200),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (147, 58, 31),
        1,
    )
    cv2.putText(
        frame,
        "vertical_ratio: " + str(gaze.vertical_ratio()),
        (90, 235),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (147, 58, 31),
        1,
    )

    cv2.imshow("Demo", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

# Clean shutdown
try:
    # 停止流
    pipeline.stop()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Allow time for cleanup
except:
    pass
