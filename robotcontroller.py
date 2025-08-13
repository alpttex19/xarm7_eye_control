import numpy as np
import mujoco
import mujoco.viewer
import time
from enum import Enum
from loguru import logger

from scipy.spatial.transform import Rotation as R

from xarm_manager_pin_casadi import CasadiRobotManager
from joystick import JoystickControl
from utils import linear_interpolation_3d


class RobotState(Enum):
    """机械臂状态枚举"""

    IDLE = 0
    GETTING_OBJECT_POSE = 1
    MOVING_TO_GRASP = 2
    GRASPING = 3
    MOVING_TO_PLACE = 4
    RELEASING = 5
    RESETTING = 6


class RobotArmController:
    def __init__(self):
        """
        初始化机械臂控制器

        Args:
            model: MuJoCo模型
            data: MuJoCo数据
            viewer: MuJoCo查看器
        """
        model = mujoco.MjModel.from_xml_path("xarm7_xml/scene.xml")
        data = mujoco.MjData(model)
        self.dof = 7  # 假设机械臂有7个自由度
        self.model = model
        self.data = data
        self.state = RobotState.IDLE

        # 手柄控制
        self.joystick_control = JoystickControl()
        # 初始化机械臂管理器
        self.robot_manager = CasadiRobotManager()

        # 控制参数
        self.gripper_open_value = 0.0  # 夹爪张开值
        self.gripper_close_value = 255.0  # 夹爪闭合值

        self.gripper_value = self.gripper_open_value  # 初始夹爪值

        # 位姿存储
        self.current_pose = np.eye(4)  # 当前末端执行器位姿
        self.last_pose = np.eye(4)  # 上次末端执行器位姿
        self.target_pose = np.eye(4)  # 目标末端执行器位姿

        # base
        self.base_pose = np.eye(4)
        self.ee_pose = np.eye(4)
        self.ee2base_pose = np.eye(4)

        self.workspace_lower = np.array([-0.7, -0.6, 0.2])  # 工作空间下界
        self.workspace_upper = np.array([0.7, 0.6, 0.8])  # 工作空间上界

        # 自动控制相关
        self.target_object = None
        self.place_position = np.array([0.1, 0, 0.25])
        self.auto_step = 0

        self.grasp_pos_offset = np.array([0.01, 0.01, 0.15])  # 抓取位姿偏移量

        # 可视化相关
        self.selected_object = None
        self.mouse_pressed = False
        self.last_click_time = 0
        self.click_threshold = 0.5  # 双击时间阈值
        self.initialize_viewer()
        self.initilize_robot_state()

    def step(self):
        # 仿真步进
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()  # 同步查看器
        time.sleep(0.001)

    def initialize_viewer(self):
        # 创建查看器
        self.viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,  # 显示左侧UI
            show_right_ui=False,  # 显示右侧UI
        )
        if self.viewer is None:
            print("查看器初始化失败")
            return False
        # 设置固定相机位置
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # 朝向
        self.viewer.cam.lookat = np.array([0.8, 0.0, 0.78])
        self.viewer.cam.distance = 1  # 相机距离
        self.viewer.cam.azimuth = 180  # 相机方位角
        self.viewer.cam.elevation = -45  # 相机仰角

    def initilize_robot_state(self):
        # 初始化机械臂关节角
        self.init_joint_pos = np.zeros(self.dof)  # 初始关节位置
        self.init_joint_pos[3] = 2.0
        self.init_joint_pos[5] = 2.0
        self.data.qpos[: self.dof] = self.init_joint_pos  # 假设前7个关节是机械臂关节
        for _ in range(10):
            self.step()  # 进行几步仿真以稳定状态

        T_home_pose = self.robot_manager.fk(self.init_joint_pos)

        self.home_pose = T_home_pose

    def initialize(self):
        """初始化机械臂到初始位置"""
        print("初始化机械臂...")
        self.state = RobotState.IDLE

        self.set_gripper(self.gripper_open_value)

        self.get_base_pose()  # 获取base位姿
        self.get_current_end_effector_pose()  # 获取末端执行器位姿

        self.move_to_pose(self.home_pose)

    def reset_all(self):
        """重置所有状态"""
        self.state = RobotState.IDLE
        self.initilize_robot_state()
        self.get_base_pose()
        self.get_current_end_effector_pose()
        self.set_gripper(self.gripper_open_value)

        # 重置手柄控制
        self.joystick_control = JoystickControl()

    def get_base_pose(self):
        # 获取body link_base的位姿
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link_base")
        if body_id >= 0:
            pos = self.data.xpos[body_id].copy()
            rot = self.data.xmat[body_id].copy()
            self.base_pose[:3, 3] = pos
            self.base_pose[:3, :3] = rot.reshape(3, 3)
            print(f"link_base位姿: {self.base_pose}")
        else:
            print("未找到link_base物体")
        return self.base_pose

    def get_current_end_effector_pose(self):
        """获取当前末端执行器位姿"""
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector"
        )
        if site_id >= 0:
            pos = self.data.site_xpos[site_id].copy()
            rot = self.data.site_xmat[site_id].copy().reshape(3, 3)
            self.ee_pose[:3, 3] = pos
            self.ee_pose[:3, :3] = rot

            self.current_pose = np.linalg.inv(self.base_pose) @ self.ee_pose
        return self.current_pose

    def move_to_pose(self, target_pose):
        """移动到目标位姿"""
        current = self.get_current_end_effector_pose()
        start_pos = current[:3, 3].copy()
        end_pos = target_pose[:3, 3].copy()
        trajectory_points = linear_interpolation_3d(start_pos, end_pos, num_points=1000)
        for point in trajectory_points:
            target_pose[:3, 3] = point

            # 使用逆运动学计算关节角度
            joint_angles = self.inverse_kinematics(target_pose)
            if joint_angles is not None:
                self.set_joint_angles(joint_angles)

            self.step()  # 执行一步仿真

    def manual_control(self):
        """手柄遥控模式"""
        pose_delta, gripper_delta = self.joystick_control.get_joystick_state()
        target_pose = self.last_pose.copy()  # 如果没有移动，保持上次位姿
        target_pose[:3, :3] = self.home_pose[:3, :3]  # 保持姿态不变

        if np.abs(np.any(pose_delta)) > 0.001:
            # 更新当前位姿
            self.current_pose = self.get_current_end_effector_pose()
            # 更新目标位姿
            target_pose[:3, 3] = self.current_pose[:3, 3]
            target_pose[:3, 3] += pose_delta

        target_pose[:3, 3] = np.clip(
            target_pose[:3, 3],
            self.workspace_lower,
            self.workspace_upper,
        )

        if np.linalg.norm(self.last_pose - target_pose) > 2:
            logger.debug(
                f"目标位姿 {target_pose[:3, 3]}, 当前位姿 {self.current_pose[:3, 3]}, 位姿增量 {pose_delta}"
            )
            logger.debug(f"目标位姿变化过大，跳过本次控制")

        # 执行运动
        joint_angles = self.inverse_kinematics(target_pose)
        if joint_angles is not None:
            self.set_joint_angles(joint_angles)

        # 夹爪控制
        self.gripper_value += gripper_delta
        self.gripper_value = np.clip(
            self.gripper_value, self.gripper_open_value, self.gripper_close_value
        )
        self.set_gripper(self.gripper_value)

        self.step()

        self.last_pose = self.current_pose.copy()

    def inverse_kinematics(self, target_pose):
        """
        逆运动学接口 - 需要根据你的实现替换

        Args:
            target_pose: 目标位姿 [x, y, z, rx, ry, rz]

        Returns:
            joint_angles: 关节角度数组
        """
        pos = target_pose[:3, 3]
        rot = target_pose[:3, :3]

        joint_angles, norm_err = self.robot_manager.ik(pos=pos, rot=rot)
        if norm_err > 0.01:
            # logger.warning(f"逆运动学误差过大: {norm_err:.4f}, 可能无法到达目标位姿")
            return None
        return joint_angles

    def set_joint_angles(self, joint_angles):
        """设置关节角度"""
        for i, angle in enumerate(joint_angles):
            if i < len(self.data.ctrl):
                self.data.ctrl[i] = angle

    def set_gripper(self, value):
        """设置夹爪状态"""
        # 假设夹爪控制器在最后一个控制输入
        gripper_id = len(self.data.ctrl) - 1
        self.data.ctrl[gripper_id] = value

    def get_object_pose(self, object_name):
        """获取指定物体的位姿"""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if body_id >= 0:
            pos = self.data.xpos[body_id].copy()
            rot = self.data.xmat[body_id].copy()
            T_obj = np.eye(4)
            T_obj[:3, 3] = pos
            T_obj[:3, :3] = rot.reshape(3, 3)

            T_obj2base = np.linalg.inv(self.base_pose) @ T_obj
            T_obj2base[:3, :3] = self.home_pose[:3, :3]
            T_obj2base[:3, 3] += self.grasp_pos_offset
            return T_obj2base
        return None

    def automatic_grasp_sequence(self, object_name, place_pos=None):
        """自动抓取序列"""
        if place_pos is not None:
            self.place_position = place_pos

        if self.state == RobotState.IDLE:
            print(f"开始抓取物体: {object_name}")
            self.target_object = object_name
            self.state = RobotState.GETTING_OBJECT_POSE
            self.auto_step = 0

    def update_automatic_control(self):
        """更新自动控制状态机"""
        if self.state == RobotState.GETTING_OBJECT_POSE:
            print("获取物体位姿...")
            object_pose = self.get_object_pose(self.target_object)
            if object_pose is not None:
                # 计算抓取位姿（在物体上方）
                grasp_pose = object_pose.copy()
                self.target_pose = grasp_pose
                self.state = RobotState.MOVING_TO_GRASP

        elif self.state == RobotState.MOVING_TO_GRASP:
            lift_pose = self.target_pose.copy()
            lift_pose[2, 3] += 0.1
            print(f"移动到抓取上方...")
            self.move_to_pose(lift_pose)

            print("移动到抓取位置...")
            self.move_to_pose(self.target_pose)

            self.state = RobotState.GRASPING

        elif self.state == RobotState.GRASPING:
            self.set_gripper(self.gripper_close_value)
            for _ in range(100):
                self.step()
            time.sleep(1.0)  # 等待夹爪闭合
            print("闭合夹爪...")

            # 向上提升
            lift_pose = self.target_pose.copy()
            lift_pose[2, 3] += 0.1
            print("提升夹爪...")
            self.move_to_pose(lift_pose)

            self.state = RobotState.MOVING_TO_PLACE

        elif self.state == RobotState.MOVING_TO_PLACE:

            place_pose = np.eye(4)
            place_pose[:3, 3] = self.place_position
            place_pose[:3, :3] = self.home_pose[:3, :3]

            lift_pose = place_pose.copy()
            lift_pose[2, 3] += 0.1
            print("移动到放置上方...")
            self.move_to_pose(lift_pose)

            print("移动到放置位置...")
            self.move_to_pose(place_pose)

            self.state = RobotState.RELEASING

        elif self.state == RobotState.RELEASING:

            print("松开夹爪...")
            self.set_gripper(self.gripper_open_value)
            for _ in range(100):
                self.step()
            time.sleep(1.0)  # 等待夹爪张开
            self.state = RobotState.RESETTING

        elif self.state == RobotState.RESETTING:
            print("复位...")
            self.move_to_pose(self.home_pose)
            self.state = RobotState.IDLE
            print("抓取任务完成！")

    def handle_click_control(self, click_pos):
        """
        处理点击控制 - 将屏幕点击转换为物体选择

        Args:
            click_pos: 点击位置 (x, y)
        """
        if self.viewer is None:
            return

        # 获取窗口尺寸
        width, height = 1280, 720  # 假设窗口大小为1280x720

        # 将屏幕坐标转换为标准化设备坐标 (NDC)
        x_ndc = (2.0 * click_pos[0] / width) - 1.0
        y_ndc = 1.0 - (2.0 * click_pos[1] / height)

        # 执行射线检测
        selected_object = self.raycast_select(x_ndc, y_ndc)
        if selected_object:
            print(f"选中物体: {selected_object}")
            self.selected_object = selected_object
            # 开始自动抓取
            self.automatic_grasp_sequence(selected_object)

    def raycast_select(self, x_ndc, y_ndc):
        """射线检测选择物体"""
        # 获取相机参数
        cam = self.viewer.cam

        # 计算射线起点和方向
        ray_start = np.zeros(3)
        ray_dir = np.zeros(3)

        # 从屏幕坐标计算射线
        mujoco.mju_ray(
            ray_start,
            ray_dir,
            x_ndc,
            y_ndc,
            self.viewer.viewport,
            cam.lookat,
            cam.distance,
            cam.azimuth,
            cam.elevation,
        )

        # 执行射线与场景的碰撞检测
        geomid = np.array([-1])
        dist = np.array([0.0])

        hit = mujoco.mj_ray(
            self.model,
            self.data,
            ray_start,
            ray_dir,
            None,
            1,  # 排除类型
            -1,  # 排除bodyid
            geomid,
            dist,
        )

        if hit and geomid[0] >= 0:
            # 获取碰撞的几何体对应的物体
            geom = self.model.geom_bodyid[geomid[0]]
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, geom)
            return body_name

        return None

    def detect_clicked_object(self, click_pos, camera_view=None):
        """检测点击的物体 - 保持向后兼容"""
        return self.raycast_select(click_pos[0], click_pos[1])

    def update(self):
        """主更新函数"""
        if self.state != RobotState.IDLE:
            self.update_automatic_control()
        self.step()


# 使用示例
def main():
    # 创建控制器
    controller = RobotArmController()
    # 初始化机械臂
    controller.initialize()

    control_type = "automatic"  # 手动控制
    if control_type == "automatic":
        # 自动控制模式
        controller.automatic_grasp_sequence("tg_L1", place_pos=[0.2, 0, 0.2])
    else:
        # 手动控制模式
        controller.reset_all()

    try:
        while controller.viewer:
            if control_type == "manual":
                controller.manual_control()
            else:
                controller.update()

    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行错误: {e}")
    finally:
        print("程序退出")


if __name__ == "__main__":
    main()
