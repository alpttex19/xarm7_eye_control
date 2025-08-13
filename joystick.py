import pygame  # 用于手柄控制
import numpy as np


class JoystickControl:

    def __init__(self):
        # 手柄控制
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"手柄已连接: {self.joystick.get_name()}")

        self.move_speed = 0.01  # 移动速度
        self.gripper_speed = 5.0  # 夹爪速度

    def get_joystick_state(self):
        """手柄遥控模式"""
        if self.joystick is None:
            print("未检测到手柄")
            return

        pygame.event.pump()

        # 读取手柄输入
        left_x = self.joystick.get_axis(0)  # 左摇杆X轴
        left_y = self.joystick.get_axis(1)  # 左摇杆Y轴
        right_y = self.joystick.get_axis(4)  # 左扳机

        right_x = self.joystick.get_axis(3)  # 右摇杆Y轴
        # 按钮
        button_a = self.joystick.get_button(0)  # A键 - 夹爪闭合
        button_b = self.joystick.get_button(1)  # B键 - 夹爪张开

        # 计算位姿增量
        pose_delta = np.zeros(3)
        pose_delta[0] = -left_y * self.move_speed  # Y轴 (反向)
        pose_delta[1] = -left_x * self.move_speed  # X轴
        pose_delta[2] = -right_y * self.move_speed  # Z轴
        for i in range(3):
            if abs(pose_delta[i]) < 0.005:
                pose_delta[i] = 0.0

        gripper_delta = right_x * self.gripper_speed  # 夹爪控制

        return pose_delta, gripper_delta
