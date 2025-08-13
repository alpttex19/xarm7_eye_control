import os
import sys
import os.path as osp
import casadi
import numpy as np
import pinocchio as pin
import time

import pinocchio.casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation as R

import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class CasadiRobotManager:
    def __init__(self):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        self.type_name = "casadi"

        self.urdf_path = osp.join(
            osp.dirname(__file__),
            "xarm7_urdf",
            "xarm7.urdf",
        )
        self.mesh_package_path = os.path.join(os.path.dirname(__file__), "xarm7_urdf")

        self.robot = pin.RobotWrapper.BuildFromURDF(
            filename=self.urdf_path, package_dirs=self.mesh_package_path
        )

        # 加载机器人模型
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        # 末端执行器Frame ID (通常是最后一个link)
        self.end_effector_frame_id = self.model.getFrameId("link7")

        self.init_data = np.zeros(self.robot.model.nq)
        self.history_data = np.zeros(self.robot.model.nq)

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # # Get the hand joint ID and define the error function
        self.gripper_id = self.robot.model.getFrameId("link7")
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.gripper_id].inverse() * cpin.SE3(self.cTf)
                    ).vector,
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.robot.model.nq)
        self.var_q_last = self.history_data  # for smooth
        self.param_tf = self.opti.parameter(4, 4)

        # 改进代价函数
        error_vec = self.error(self.var_q, self.param_tf)
        self.totalcost = casadi.dot(error_vec, error_vec)  # 使用dot而不是sumsqr
        self.regularization = casadi.dot(self.var_q, self.var_q)

        # Setting optimization constraints and goals
        self.opti.subject_to(
            self.opti.bounded(
                np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]),
                self.var_q,
                np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]),
            )
        )
        self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)

        opts = {
            "ipopt": {"print_level": 1, "max_iter": 1000, "tol": 1e-3},
            "print_time": False,
        }
        self.opti.solver("ipopt", opts)

    def ik_solver(self, target_pose, motorstate=None, motorV=None):
        if motorstate is not None:
            self.init_data = motorstate
        self.opti.set_initial(self.var_q, self.init_data)

        self.opti.set_value(self.param_tf, target_pose)
        # self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            # sol = self.opti.solve()
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)

            # total_cost

            if self.init_data is not None:
                max_diff = max(abs(self.history_data - sol_q))
                # print("max_diff:", max_diff)
                self.init_data = sol_q
                if max_diff > 30.0 / 180.0 * 3.1415:
                    # print("Excessive changes in joint angle:", max_diff)
                    self.init_data = np.zeros(self.robot.model.nq)
            else:
                self.init_data = sol_q
            self.history_data = sol_q

            if motorV is not None:
                v = motorV * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            tau_ff = pin.rnea(
                self.robot.model,
                self.robot.data,
                sol_q,
                v,
                np.zeros(self.robot.model.nv),
            )

            return sol_q, tau_ff

        except Exception as e:
            # print(f"ERROR in convergence, plotting debug info.{e}")
            sol_q = self.opti.debug.value(self.var_q)  # return original value
            return sol_q, []

    def ik(self, pos=np.zeros(3), rot=np.eye(3), last_q=None):
        target_se3 = pin.SE3(rot, pos)
        sol_q, tau_ff = self.ik_solver(target_se3.homogeneous)

        # casadi fk
        res_error = self.error(sol_q, target_se3.homogeneous)

        norm_err = np.linalg.norm(res_error)

        return sol_q, norm_err

    def fk(self, q):
        # 更新Pinocchio模型
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # 获取末端执行器位姿
        T_matrix = self.data.oMf[self.end_effector_frame_id].homogeneous

        return T_matrix


def test_ik():
    ik_solver = CasadiRobotManager()
    q_init = np.zeros(7)
    q_init[3] = 1.57
    T_fk = ik_solver.fk(q_init)

    rot = T_fk[:3, :3]
    pos = T_fk[:3, 3]

    q_solved, error_norm = ik_solver.ik(pos=pos, rot=rot)

    T_ik = ik_solver.fk(q_solved)
    print("FK Result:")
    print(T_ik)
    print("Target Pose:")
    print(T_fk)
    print("Solved Joint Angles:")
    print(q_solved, error_norm)


if __name__ == "__main__":
    test_ik()
