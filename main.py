import mujoco
import threading
import time
from enum import Enum
import numpy as np
import cv2
from GazeTracking.gaze_tracking import GazeTracking
import os

from robotcontroller import RobotArmController

try:
    import pyrealsense2 as rs
except ImportError:
    print("pyrealsense2 module not found. using webcam.")
    rs = None

camera_id = 4  # 默认摄像头ID，可能需要根据实际情况调整


class ControlMode(Enum):
    MANUAL = "manual"
    AUTO = "auto"  # 新增自动眼部追踪模式
    AUTO_GRASP_L1 = "auto_l1"
    AUTO_GRASP_L2 = "auto_l2"
    AUTO_GRASP_M = "auto_m"
    RESET = "reset"
    PAUSE = "pause"


class EyeGazeTracker:
    """眼部注视追踪器"""

    def __init__(self, threshold_count=30):
        # 设置环境变量以避免Qt线程问题
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

        # 初始化CV2
        cv2.setUseOptimized(True)
        try:
            cv2.setLogLevel(0)
        except:
            pass

        # 配置 RealSense 管道
        if rs:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        else:
            self.webcam = cv2.VideoCapture(camera_id)

        # 初始化注视追踪
        self.gaze = GazeTracking()

        # 计数器和阈值
        self.threshold_count = threshold_count
        self.right_count = 0
        self.center_count = 0
        self.left_count = 0

        # 状态标志
        self.is_running = False
        self.selected_option = None

    def start_tracking(self):
        """开始眼部追踪"""
        if self.is_running:
            return
        # 创建窗口
        cv2.namedWindow("Gaze Tracking", cv2.WINDOW_NORMAL)

        try:
            if rs:
                self.pipeline.start(self.config)
            self.is_running = True
            self.reset_counters()
            print("👁️ 眼部追踪已启动")
            print(f"注视阈值设置为: {self.threshold_count} 次")
            print("请注视相应方向来选择操作:")
            print("  👉 右看 -> 选择1 (红色方块)")
            print("  👀 中看 -> 选择2 (蓝色方块)")
            print("  👈 左看 -> 选择3 (黄色方块)")
        except Exception as e:
            print(f"启动眼部追踪失败: {e}")
            self.is_running = False

    def stop_tracking(self):
        """停止眼部追踪"""
        if not self.is_running:
            return

        try:
            if rs:
                self.pipeline.stop()
            else:
                self.webcam.release()
            cv2.destroyAllWindows()
            self.is_running = False
            print("👁️ 眼部追踪已停止")
        except Exception as e:
            print(f"停止眼部追踪时出错: {e}")

    def reset_counters(self):
        """重置计数器"""
        self.right_count = 0
        self.center_count = 0
        self.left_count = 0
        self.selected_option = None

    def update(self):
        """更新追踪状态"""
        if not self.is_running:
            return None

        try:
            if rs:
                # 获取帧数据
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                if not color_frame:
                    return None

                # 转换为numpy数组
                frame = np.asanyarray(color_frame.get_data())
            else:
                # 从摄像头获取帧
                ret, frame = self.webcam.read()
                if not ret:
                    return None

            # 发送帧到注视追踪进行分析
            self.gaze.refresh(frame)

            # 获取标注后的帧
            frame = self.gaze.annotated_frame()

            # 检测注视方向并更新计数器
            direction_text = ""
            if self.gaze.is_blinking():
                direction_text = "Blinking"
            elif self.gaze.is_right(threshold=0.55):
                self.right_count += 1
                direction_text = f"Looking right ({self.right_count})"
            elif self.gaze.is_left(threshold=0.65):
                self.left_count += 1
                direction_text = f"Looking left ({self.left_count})"
            elif self.gaze.is_center():
                self.center_count += 1
                direction_text = f"Looking center ({self.center_count})"

            # 检查是否达到阈值
            if (
                self.right_count >= self.threshold_count
                and self.selected_option is None
            ):
                self.selected_option = 1
                print(f"✅ 检测到右看 {self.right_count} 次，选择选项1 (红色方块)")
            elif (
                self.center_count >= self.threshold_count
                and self.selected_option is None
            ):
                self.selected_option = 2
                print(f"✅ 检测到中看 {self.center_count} 次，选择选项2 (蓝色方块)")
            elif (
                self.left_count >= self.threshold_count and self.selected_option is None
            ):
                self.selected_option = 3
                print(f"✅ 检测到左看 {self.left_count} 次，选择选项3 (黄色方块)")

            # 在帧上添加文本信息
            cv2.putText(
                frame,
                direction_text,
                (50, 50),
                cv2.FONT_HERSHEY_DUPLEX,
                1.2,
                (0, 255, 0),
                2,
            )

            # 显示计数器
            counter_text = f"Right: {self.right_count}, Center: {self.center_count}, Left: {self.left_count}"
            cv2.putText(
                frame,
                counter_text,
                (50, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # 显示阈值信息
            threshold_text = f"Threshold: {self.threshold_count}"
            cv2.putText(
                frame,
                threshold_text,
                (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # 如果有选择，显示选择结果
            if self.selected_option:
                selection_text = f"Selected: Option {self.selected_option}"
                cv2.putText(
                    frame,
                    selection_text,
                    (50, 150),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                )

            # 显示瞳孔坐标等详细信息
            left_pupil = self.gaze.pupil_left_coords()
            right_pupil = self.gaze.pupil_right_coords()
            cv2.putText(
                frame,
                f"Left pupil: {left_pupil}",
                (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (147, 58, 31),
                1,
            )
            cv2.putText(
                frame,
                f"Right pupil: {right_pupil}",
                (50, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (147, 58, 31),
                1,
            )

            # 显示帧
            cv2.imshow("Gaze Tracking", frame)
            cv2.waitKey(1)

            return self.selected_option

        except Exception as e:
            print(f"更新眼部追踪时出错: {e}")
            return None


class KeyboardControlledRobot:
    def __init__(self):
        self.controller = RobotArmController()
        self.eye_tracker = EyeGazeTracker(threshold_count=20)  # 可以调整阈值

        # 控制状态
        self.current_mode = ControlMode.PAUSE
        self.previous_mode = ControlMode.PAUSE
        self.running = True
        self.mode_changed = False

        # 任务配置
        self.grasp_tasks = {
            ControlMode.AUTO_GRASP_L1: {
                "target": "tg_L1",
                "place_pos": [0.15, 0.07, 0.2],
            },
            ControlMode.AUTO_GRASP_L2: {
                "target": "tg_L2",
                "place_pos": [0.2, 0.0, 0.2],
            },
            ControlMode.AUTO_GRASP_M: {
                "target": "tg_M",
                "place_pos": [0.25, -0.07, 0.2],
            },
        }

        # 启动键盘监听线程
        self.keyboard_thread = threading.Thread(
            target=self.keyboard_listener, daemon=True
        )
        self.keyboard_thread.start()

    def keyboard_listener(self):
        """键盘监听线程"""
        print("\n=== 机械臂控制模式 ===")
        print("键盘控制说明:")
        print("  M - 手动控制模式")
        print("  A - 自动眼部追踪模式")  # 修改了这里的说明
        print("  1 - 自动抓取 tg_L1 (红色方块)")
        print("  2 - 自动抓取 tg_L2 (蓝色方块)")
        print("  3 - 自动抓取 tg_M (黄色方块)")
        print("  R - 重置机械臂")
        print("  P - 暂停/继续")
        print("  H - 显示帮助")
        print("  Q - 退出程序")
        print("========================\n")

        while self.running:
            try:
                key = input().strip().upper()
                old_mode = self.current_mode

                if key == "M":
                    self.current_mode = ControlMode.MANUAL
                elif key == "A":
                    self.current_mode = ControlMode.AUTO  # 切换到自动眼部追踪模式
                elif key == "1":
                    self.current_mode = ControlMode.AUTO_GRASP_L1
                elif key == "2":
                    self.current_mode = ControlMode.AUTO_GRASP_L2
                elif key == "3":
                    self.current_mode = ControlMode.AUTO_GRASP_M
                elif key == "R":
                    self.current_mode = ControlMode.RESET
                elif key == "P":
                    if self.current_mode == ControlMode.PAUSE:
                        self.current_mode = self.previous_mode
                        print("▶️ 继续运行")
                    else:
                        self.previous_mode = self.current_mode
                        self.current_mode = ControlMode.PAUSE
                        print("⏸️ 已暂停")
                elif key == "H":
                    self.show_help()
                elif key == "Q":
                    print("正在退出程序...")
                    self.running = False
                    break
                else:
                    print(f"未知命令: {key}")
                    continue

                # 检查模式是否改变
                if old_mode != self.current_mode:
                    self.mode_changed = True
                    if key != "P":  # 暂停命令不显示模式切换信息
                        print(f"🔄 模式切换: {self.get_mode_description()}")

            except EOFError:
                # 处理Ctrl+D
                break
            except Exception as e:
                print(f"键盘输入错误: {e}")

    def get_mode_description(self):
        """获取当前模式的描述"""
        descriptions = {
            ControlMode.MANUAL: "手动控制模式",
            ControlMode.AUTO: "自动眼部追踪模式",
            ControlMode.AUTO_GRASP_L1: "自动抓取红色方块 (tg_L1)",
            ControlMode.AUTO_GRASP_L2: "自动抓取蓝色方块 (tg_L2)",
            ControlMode.AUTO_GRASP_M: "自动抓取黄色方块 (tg_M)",
            ControlMode.RESET: "重置机械臂",
            ControlMode.PAUSE: "暂停模式",
        }
        return descriptions.get(self.current_mode, "未知模式")

    def show_help(self):
        """显示帮助信息"""
        print("\n=== 帮助信息 ===")
        print(f"当前模式: {self.get_mode_description()}")
        print("可用命令:")
        print("  M - 切换到手动控制")
        print("  A - 切换到自动眼部追踪模式")
        print("  1/2/3 - 自动抓取对应方块")
        print("  R - 重置机械臂到初始位置")
        print("  P - 暂停/继续当前操作")
        print("  H - 显示此帮助")
        print("  Q - 退出程序")
        print("=================\n")

    def execute_current_mode(self):
        """执行当前模式的操作"""
        try:
            if self.current_mode == ControlMode.MANUAL:
                self.controller.manual_control()

            elif self.current_mode == ControlMode.AUTO:
                # 自动眼部追踪模式
                if not self.eye_tracker.is_running:
                    self.eye_tracker.start_tracking()

                # 更新眼部追踪
                selected_option = self.eye_tracker.update()

                if selected_option:
                    print(f"🎯 眼部追踪选择了选项 {selected_option}")

                    # 停止眼部追踪
                    self.eye_tracker.stop_tracking()

                    # 根据选择执行相应的抓取任务
                    if selected_option == 1:
                        self.current_mode = ControlMode.AUTO_GRASP_L1
                    elif selected_option == 2:
                        self.current_mode = ControlMode.AUTO_GRASP_L2
                    elif selected_option == 3:
                        self.current_mode = ControlMode.AUTO_GRASP_M

                    self.mode_changed = True
                    print(f"🔄 自动切换到: {self.get_mode_description()}")

                # 更新控制器显示
                self.controller.update()

            elif self.current_mode in self.grasp_tasks:
                task = self.grasp_tasks[self.current_mode]
                print(f"🤖 开始执行: {self.get_mode_description()}")
                success = self.controller.automatic_grasp_sequence(
                    task["target"], place_pos=task["place_pos"]
                )
                if success:
                    print(f"✅ 任务完成: {self.get_mode_description()}")
                else:
                    print(f"❌ 任务失败: {self.get_mode_description()}")
                # 任务完成后切换到暂停模式
                self.previous_mode = self.current_mode
                self.current_mode = ControlMode.PAUSE

            elif self.current_mode == ControlMode.RESET:
                print("🔄 重置机械臂...")
                self.controller.reset_all()
                print("✅ 重置完成")
                # 重置后切换到暂停模式
                self.current_mode = ControlMode.PAUSE

            elif self.current_mode == ControlMode.PAUSE:
                # 暂停模式只更新显示
                self.controller.update()

        except Exception as e:
            print(f"执行模式时出错: {e}")
            self.current_mode = ControlMode.PAUSE

    def run(self):
        """主运行循环"""
        # 初始化机械臂
        self.controller.initialize()

        try:
            while self.running and self.controller.viewer:
                # 检查模式是否改变
                if self.mode_changed:
                    self.mode_changed = False
                    # 如果切换到非AUTO模式，停止眼部追踪
                    if (
                        self.current_mode != ControlMode.AUTO
                        and self.eye_tracker.is_running
                    ):
                        self.eye_tracker.stop_tracking()

                # 执行当前模式
                self.execute_current_mode()

                # 小延时避免CPU占用过高
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"程序运行错误: {e}")
        finally:
            # 清理资源
            if self.eye_tracker.is_running:
                self.eye_tracker.stop_tracking()
            self.running = False
            print("程序退出")


def main():
    """主函数"""

    try:
        print("正在加载模型...")
        robot = KeyboardControlledRobot()
        print("✅ 模型加载成功")

        # 运行控制循环
        robot.run()

    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        print("请确保模型文件路径正确")
        return


if __name__ == "__main__":
    main()
