import os
from pathlib import Path

import numpy as np
import placo
from spatialmath import SE3

from .robot import Robot


class RobotWrapper(Robot):
    def __init__(self) -> None:
        super().__init__()

        self._dof = 7

        urdf_dir = os.path.join(
            Path(__file__).parent.parent.parent,
            "assets/ufactory_xarm7/xarm7_urdf/xarm7.urdf",
        )
        self._robot_wrapper = placo.RobotWrapper(
            os.fspath(urdf_dir), placo.Flags.ignore_collisions
        )

        self._joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

        self._solver = placo.KinematicsSolver(self._robot_wrapper)
        self._solver.mask_fbase(True)
        self._effector_task = self._solver.add_frame_task("link7", np.eye(4))
        self._effector_task.configure("link7", "soft", 1.0, 0.1)

        # self._manipulability = self._solver.add_manipulability_task(
        #     "link7", "both", 1.0
        # )
        # self._manipulability.configure("manipulability", "soft", 1e-1)

        # self._solver.enable_joint_limits(True)
        # self._solver.enable_velocity_limits(True)
        # self._solver.dt = 0.01

    def fkine(self, q) -> SE3:
        for i, joint_name in enumerate(self._joint_names):
            self._robot_wrapper.set_joint(joint_name, q[i])
        self._robot_wrapper.update_kinematics()
        return (
            self._base
            * SE3(self._robot_wrapper.get_T_world_frame("link7"))
            * self._tool
        )

    def ikine(self, Twt: SE3, max_iterations=100) -> np.ndarray:
        """
        改进的IK求解方法
        """
        # 目标变换矩阵
        T_target = (self._base.inv() * Twt * self._tool.inv()).A
        self._effector_task.T_world_frame = T_target

        best_q = np.zeros(self._dof)
        best_error = float("inf")

        # 多次尝试求解
        for attempt in range(max_iterations):
            try:
                # 求解
                result = self._solver.solve(True)
                self._robot_wrapper.update_kinematics()
                # self._solver.dump_status()

                # 检查收敛性
                current_T = self._robot_wrapper.get_T_world_frame("link7")
                pos_error = np.linalg.norm(T_target[:3, 3] - current_T[:3, 3])
                rot_error = np.linalg.norm(T_target[:3, :3] - current_T[:3, :3])
                total_error = pos_error + rot_error

                # 保存最佳解
                if total_error < best_error:
                    best_error = total_error
                    for i, joint_name in enumerate(self._joint_names):
                        best_q[i] = self._robot_wrapper.get_joint(joint_name)

                # 如果误差足够小，提前退出
                if pos_error < 1e-4 and rot_error < 1e-3:
                    break

            except Exception as e:
                print(f"IK attempt {attempt} failed: {e}")
                # 如果求解失败，稍微扰动初始值再试
                if attempt < max_iterations - 1:
                    noise = np.random.normal(0, 0.01, 7)
                    q_perturbed = self._clip_to_joint_limits(q_init + noise)
                    for i, joint_name in enumerate(self._joint_names):
                        self._robot_wrapper.set_joint(joint_name, q_perturbed[i])

        return best_q

    def set_joint(self, q):
        super().set_joint(q)
        for i, joint_name in enumerate(self._joint_names):
            self._robot_wrapper.set_joint(joint_name, q[i])
        self._robot_wrapper.update_kinematics()

    def set_base(self, base: SE3):
        self._base = base.copy()

    def set_tool(self, tool: SE3):
        self._tool = tool.copy()

    def disable_base(self):
        self._base = SE3()

    def disable_tool(self):
        self._tool = SE3()

    def test_fk_ik(self, num_tests=5):
        """
        简单的FK-IK测试：随机关节角 -> FK -> IK -> FK
        """
        print("Testing FK-IK consistency...")
        print("=" * 40)

        self.disable_base()
        self.disable_tool()

        success_count = 0

        for i in range(num_tests):
            # 生成随机关节角（在合理范围内）
            q_original = np.random.uniform(-np.pi / 2, np.pi / 2, 7)

            print(f"\nTest {i+1}:")
            print(f"Original q: {q_original}")

            # 步骤1: FK - 计算末端位姿
            T_target = self.fkine(q_original)

            # 步骤2: IK - 从末端位姿求解关节角
            q_solved = self.ikine(T_target)
            print(f"IK solved q: {q_solved}")

            print(f"FK result position: [{T_target}]")
            # 步骤3: FK验证 - 用求解的关节角再次计算末端位姿
            T_verify = self.fkine(q_solved)
            print(f"Verify position: [{T_verify}]")

            # 计算位置误差
            pos_error = np.linalg.norm(T_target.t - T_verify.t)
            print(f"Position error: {pos_error:.6f} m")

            # 判断是否成功
            if pos_error < 0.001:  # 1mm误差容忍度
                print("✅ PASS")
                success_count += 1
            else:
                print("❌ FAIL")

        print(f"\n{'='*40}")
        print(f"Test Summary: {success_count}/{num_tests} tests passed")
        print(f"Success rate: {100*success_count/num_tests:.1f}%")

        return success_count == num_tests

    def test_fk_basic(self, test_configs=None):
        """
        基本FK测试：测试几个固定配置
        """
        print("Testing basic FK...")
        print("=" * 30)

        if test_configs is None:
            test_configs = [
                np.zeros(7),  # 零位
                np.array([0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7]),  # 随机1
                np.array([np.pi / 4, 0, 0, 0, 0, 0, 0]),  # 第一关节45度
            ]

        self.disable_base()
        self.disable_tool()

        for i, q in enumerate(test_configs):
            print(f"\nTest {i+1}: q = {q}")

            try:
                T = self.fkine(q)
                pos = T.t
                R = T.R

                print(f"Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
                print(f"Rotation matrix det: {np.linalg.det(R):.6f}")

                # 检查旋转矩阵是否有效
                if abs(np.linalg.det(R) - 1.0) < 1e-6:
                    print("✅ Valid rotation matrix")
                else:
                    print("❌ Invalid rotation matrix")

            except Exception as e:
                print(f"❌ Error: {str(e)}")

    def quick_test(self):
        """
        快速测试FK和IK
        """
        print("🤖 Quick Robot Test")
        print("=" * 20)

        # 测试基本FK
        self.test_fk_basic()

        print("\n")

        # 测试FK-IK一致性
        return self.test_fk_ik(num_tests=100)


if __name__ == "__main__":
    robot = RobotWrapper()
    robot.quick_test()
