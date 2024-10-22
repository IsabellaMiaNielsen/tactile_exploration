#Admittance Controller

import numpy as np
from scipy.spatial.transform import Rotation
from utils import utility
import time
from spatialmath import SE3, SO3
from spatialmath.base import q2r, r2q
import math

EPS = np.finfo(float).eps * 4.0


class Admittance:
    def __init__(self, robot, m, d):

        '''
        INPUTS:
            target_force -> force to push against object (around 4N)
            wrench -> the forces exerted by the surface on the end effector due to contact. Sensor reading (3 element vector, [Fx; Fy; Fz])
            des_vel -> desired velocity during movement. VALUE???
            rot_align -> rotation matrix the robot should have to be aligned with surface normal
            Xc -> current_pos 
            current_vel
            target_force, initial_wrench, current_TCP, model_data, dt

        RETUNS:
            Xc = desired end effector position based on controller response to external force (surface normal) in TOOL FRAME
        '''

        self.robot = robot
        self.model = m
        self.data = d

        self.dt = 0.002
        
        self.target_tol = 0.008 #0.0075

        # Gain matrices
        m = 100
        kenv = 20000
        kd = 300 # 1
        k = 10 #4/m * kd - kenv

        self.M = np.array([[m,0,0],[0,m,0],[0,0,m]])
        self.K = np.array([[k,0,0],[0,k,0],[0,0,0]])
        self.D = np.array([[kd,0,0],[0,kd,0],[0,0,kd]])

        #Initial conditions:
        self._x_d = np.array([0.0, 0.0, 0.0])
        self._dc_c = np.array([0.0, 0.0, 0.0])
        self._x_e = np.array([0.0, 0.0, 0.0])
        self._dx_e = np.array([0.0, 0.0, 0.0])
        self.target_force = np.array([0.0, 0.0, 0.0])

        self.target = self.robot.get_ee_pose().t
        self.force = np.array([0, 0, 0])


    def admittance(self):
        tcp_pose = self.robot.get_ee_pose()
        tcp_rot_mat = tcp_pose.R
        tcp_pos = tcp_pose.t

        tcp_quat = r2q(tcp_rot_mat, order='xyzs')

        self.force = np.array([0, 0, 0])
        self.force_memory = np.array([0, 0, 0])

        # Check for contact and get current force
        self.force, rot_contact, is_in_contact = utility._get_contact_info(model=self.model, data=self.data, actor='gripper', obj='pikachu')

        if np.max(np.abs(self.force)) < self.target_tol:
            self.force = self.force_memory

        self.force_memory = self.force

        if is_in_contact:
            # self.target_force = np.array([0.0, 0.0, -2.0]) # Direction in base-frame
            target_force_frame = SE3.Rt(tcp_pose.R, [1,1,1]) * SE3.Rt(np.eye(3), [0.0, 0.0, 14.0]) # Direction in base-frame
            self.target_force = target_force_frame.t

            r = utility.directionToNormal(
                    tcp_pose.R,
                    self.force, 
                    rot=rot_contact
                    )
            self.align_rot_matrix = r.as_matrix()
            self.target[-4:] = r2q(np.asarray(self.align_rot_matrix), order='xyzs') 

            # print("Current Pose: ", tcp_pose.R)
            # print("Concact Pose: ", rot_contact)
            # print("Align Pose: ", self.align_rot_matrix)
        else:
            self.align_rot_matrix = q2r(self.target[-4:], order='xyzs')

        print("Target force: ", self.target_force)

        self.actual_pose = np.concatenate([tcp_pos, tcp_quat])
        self.target_pose = self.target
        print("Actual pose: ,", self.actual_pose)
        print("Target pose: ", self.target_pose)

        self._x_d = self.target[:3]
        
        # Update gains based on orientation function
        # self.M = self.align_rot_matrix @ self.M
        # self.K = self.align_rot_matrix @ self.K
        # self.D = self.align_rot_matrix @ self.D


        # Positional part of the admittance controller
        # Step 1: Acceleration error
        self._ddx_e = np.linalg.inv(self.M) @ (-self.force + self.target_force - self.K @ self._x_e - self.D @ self._dx_e)

        # Step 2: Integrate -> velocity error
        self._dx_e += self._ddx_e * self.dt # Euler integration

        # Step 3: Integrate -> position error
        self._x_e += self._dx_e * self.dt # Euler integration

        # Step 4: Update the position
        self._x_c = self._x_d + self._x_e


        # print("Current Position: ", tcp_pos)
        # print("Desired Position: ", self._x_d)
        # print("Force: ", -force)
        # print("Position error: ", self._x_e)
        # print("Compliant Position: ", self._x_c)

        # compliant_pose = SE3.Rt(np.asarray(self.align_rot_matrix), self._x_c)
        # print("TCP: ", SE3.Rt(tcp_pose.R, self._x_c))

        if max(np.abs(self.actual_pose[-4:] - self.target_pose[-4:])) < self.target_tol:
            compliant_pose = SE3.Rt(np.asarray(self.align_rot_matrix), self._x_c)
        else:
            compliant_pose = utility.get_ee_transformation(tcp_pose.R, tcp_pose.t, np.asarray(self.align_rot_matrix))
        return compliant_pose


    def target_reached(self):
        if self.actual_pose is not None and self.target_pose is not None:
            return max(np.abs(self.actual_pose - self.target_pose)) < self.target_tol
        else:
            return False
        

    def step(self):
        compliant_pose = self.admittance()

        # Move robot to compliant pose
        self.robot.set_ee_pose(compliant_pose) 
        
        return self.target_reached()
