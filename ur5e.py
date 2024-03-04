import mujoco as mj
import roboticstoolbox as rtb
import spatialmath as sm
import math as m
import numpy as np
import os
import warnings
import sys


from typing import List, Union, Dict

from spatialmath import SE3

from utils.mj import (
    get_actuator_names,
    get_joint_names,
    get_joint_value,
    get_actuator_value,
    set_actuator_value,
    is_done_actuator,
    get_joint_range,
    set_joint_value,
    set_joint_qpos0
)

from utils.rtb import (
    make_tf
)

from utils.sim import (
    read_config, 
    save_config,
    config_to_q,
    RobotConfig
)

class UR5e:
    def __init__(self, model: mj.MjModel, data: mj.MjData, args, init_q:list = []) -> None:
        super().__init__()

        self._args = args

        self._UR_TCP_TO_HAND_E_TCP = 0.16

        self._robot = rtb.DHRobot(
            [
                rtb.RevoluteDH(d = 0.1625, alpha = m.pi / 2.0  , qlim=(-m.pi,m.pi)), # J1
                rtb.RevoluteDH(a = -0.425                     , qlim=(-m.pi,m.pi)), # J2
                rtb.RevoluteDH(a = -0.3922                    , qlim=(-m.pi,m.pi)), # J3
                rtb.RevoluteDH(d = 0.1333, alpha =  m.pi / 2.0, qlim=(-m.pi,m.pi)), # J4
                rtb.RevoluteDH(d = 0.0997, alpha = -m.pi / 2.0, qlim=(-m.pi,m.pi)), # J5
                rtb.RevoluteDH(d = 0.0996 + self._UR_TCP_TO_HAND_E_TCP                    , qlim=(-m.pi,m.pi)), # J6
            ], name=self.name, base=SE3.Rz(m.pi)                                     # base transform due to fkd ur standard
            # ], name=self.name, base=SE3.Rz(m.pi)                                     # base transform due to fkd ur standard
        )

        self._model = model
        self._data = data
        self._actuator_names = self._get_actuator_names()
        self._joint_names    = self._get_joint_names()
        self._config_dir     = self._get_config_dir()
        self._configs        = self._get_configs()

        self._ik_solution_pool = self._args.ik_solution_pool
        self._robot_handle = None

        if len(init_q) != 0:
            for i, jn in enumerate(self._joint_names):
                set_joint_qpos0(model=self._model,joint_name=jn, qpos0=init_q[i])
                set_actuator_value(data=self._data,actuator_name=self._actuator_names[i],q=init_q[i])

    @property
    def args(self):
        return self._args

    @property
    def mj_data(self) -> mj.MjData:
        return self._data
    
    @property
    def mj_model(self) -> mj.MjModel:
        return self._model

    @property
    def name(self) -> str:
        return "ur5e"

    @property
    def n_actuators(self) -> int:
        return len(self._actuator_names)

    @property
    def n_joints(self) -> int:
        return len(self._joint_names)

    @property
    def actuator_values(self) -> list[float]:
        return [ get_actuator_value(self._data, an) for an in self._actuator_names ]


    def _set_robot_handle(self, robot_handle):
        self._robot_handle = robot_handle

    def _config_to_q(self, config: str) -> List[float]:
        return config_to_q(
                    cfg = config, 
                    configs = self._configs, 
                    actuator_names = self._actuator_names
                )

    def set_q(self, q : Union[str, List, RobotConfig]):
        if isinstance(q, str):
            q: List[float] = self._config_to_q(config=q)
        if isinstance(q, RobotConfig):
            q: List[float] = q.joint_values
        assert len(q) == self.n_actuators, f"Length of q should be {self.n_actuators}, q had length {len(q)}"


        if len(self._robot_handle._traj) == 0:
            qf = self._robot_handle.get_q().joint_values
        else:
            qf = self._robot_handle._traj[-1].copy()

        # qf = self._robot_handle._traj[-1].copy()
        qf[:self.n_actuators] = self._clamp_q(q)

        self._robot_handle._traj.extend([qf])

    def set_ee_pose(self, 
            pos: List = [0.5,0.5,0.5], 
            ori: Union[np.ndarray,SE3] = [1,0,0,0], 
            pose: Union[None, List[float], np.ndarray, SE3] = None,
            ) -> None:

        if pose is not None:
            if isinstance(pose, SE3):
                target_pose = pose
            else:
                # Assuming pose is a list or numpy array [x, y, z, qw, qx, qy, qz]
                target_pose = SE3(pose[:3], pose[3:])
        else:
            # Use the provided position and orientation
            target_pose = make_tf(pos=pos, ori=ori)

        q_sols = []
        for _ in range(self._ik_solution_pool):
            q_sol, success, iterations, searches, residual = self._robot.ik_NR(Tep=target_pose)
            q_sols.append( (q_sol, success, iterations, searches, residual) )
        
        if not np.any([ x[1] for x in q_sols ]):
            raise ValueError(f"Inverse kinematics failed to find a solution to ee pose. Tried {self._ik_solution_pool} times. [INFO]: \n\t{q_sol=}\n\t{success=}\n\t{iterations=}\n\t{searches=}\n\t{residual=}")

        if len(self._robot_handle._traj) == 0:
            q0 = self._robot_handle.get_q().joint_values[:self.n_actuators]
        else:
            q0 = self._robot_handle._traj[-1].copy()[:self.n_actuators]
            # q0 = self._robot_handle._traj[-1].copy()[:self.n_actuators]

        lengths = []
        for i in range(len(q_sols)):
            q_end = q_sols[i][0]
            # traj = rtb.jtraj(q0 = q0, qf = q_end, t = 50)
            lenght = np.linalg.norm(q_end - q0)
            lengths.append((lenght,i))

        best_i = min(lengths)[1]

        if len(self._robot_handle._traj) == 0:
            q_robot = self._robot_handle.get_q().joint_values
        else:
            q_robot = self._robot_handle._traj[-1].copy()
        # q_robot = self._robot_handle._traj[-1].copy()

        q_robot[:self.n_actuators] = self._clamp_q(q_sols[best_i][0])

        q_robot = q_robot if isinstance(q_robot,list) else q_robot.tolist()

        self._robot_handle._traj.extend([ q_robot ])

    def get_ee_pose(self) -> SE3:
        return SE3(self._robot.fkine(self.get_q().joint_values))