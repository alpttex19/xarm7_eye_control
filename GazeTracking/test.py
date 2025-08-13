import cv2
import numpy as np
import pyrealsense2 as rs

# 配置 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()

# 启用彩色和深度流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# 开始流
pipeline.start(config)

try:
    while True:
        # 等待一帧数据
        frames = pipeline.wait_for_frames()

        # 获取深度帧和彩色帧
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # 将帧转换为 numpy 数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 应用深度图的色彩映射
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        # 水平拼接图像
        # images = np.hstack((color_image, depth_colormap))

        # 显示图像
        cv2.imshow("RealSense Color and Depth", color_image)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # 停止流
    pipeline.stop()
    cv2.destroyAllWindows()
