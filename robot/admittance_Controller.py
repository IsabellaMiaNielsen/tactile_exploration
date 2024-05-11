#Admittance Controller

import numpy as np
from scipy.spatial.transform import Rotation
from utils import utility
import time
from spatialmath import SE3, SO3
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

        self.dt = 0.001
        
        self.target_tol = 0.0075 #0.0075

        # Gain matrices
        m = 1
        kenv = 20000 # 5000 for softbody
        kd = 250 # 1
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

        self.object_center = [0.4, 0.2, 0.08]

        self.aligned = False


    def admittance(self, target):
        self.target = np.copy(target)
        tcp_pose = self.robot.get_ee_pose()
        tcp_rot_mat = tcp_pose.R
        tcp_pos = tcp_pose.t

        tcp_quat = self.mat2quat(tcp_rot_mat)

        # Check for contact and get current force
        force, rot_contact, is_in_contact = utility._get_contact_info(model=self.model, data=self.data, actor='gripper', obj='pikachu')


        if is_in_contact:
            self.target_force = np.array([0.0, 0.0, -2.0]) # Direction in base-frame

            r = utility.directionToNormal(
                    tcp_pose.R,
                    force, 
                    rot=rot_contact
                    )
            align_rot_matrix = r.as_matrix()
            self.target[-4:] = r.as_quat()
        else:
            align_rot_matrix = self.quat2mat(self.target[-4:])


        self.actual_pose = np.concatenate([tcp_pos, tcp_quat])
        self.target_pose = self.target
        print("Actual pose: ,", self.actual_pose)
        print("Target pose: ", self.target_pose)

        self._x_d = self.target[:3]

        
        # Update gains based on orientation function
        # self.M = rot_align @ self.M
        # self.K = rot_align @ self.K
        # self.D = rot_align @ self.D


        # Positional part of the admittance controller
        # Step 1: Acceleration error
        self._ddx_e = np.linalg.inv(self.M) @ (-force + self.target_force - self.K @ self._x_e - self.D @ self._dx_e)

        # Step 2: Integrate -> velocity error
        self._dx_e += self._ddx_e * self.dt # Euler integration

        # Step 3: Integrate -> position error
        self._x_e += self._dx_e * self.dt # Euler integration

        # Step 4: Update the position
        self._x_c = self._x_d + self._x_e
        

        print("Current Position: ", tcp_pos)
        print("Desired Position: ", self._x_d)
        print("Force: ", -force)
        # print("Position error: ", self._x_e)
        print("Compliant Position: ", self._x_c)

        compliant_pose = SE3.Rt(align_rot_matrix, self._x_c)

        print("TCP: ", SE3.Rt(tcp_pose.R, self._x_c))

        # compliant_pose = utility.get_ee_transformation(tcp_pose.R, tcp_pose.t, align_rot_matrix) 
        return compliant_pose


    def target_reached(self):
        if self.actual_pose is not None and self.target_pose is not None:
            print(max(np.abs(self.actual_pose - self.target_pose)))
            return max(np.abs(self.actual_pose - self.target_pose)) < self.target_tol
        else:
            return False
        

    def step(self, target):
        compliant_pose = self.admittance(target)

        # Move robot to compliant pose
        self.robot.set_ee_pose(compliant_pose) 
        
        return self.target_reached()







    def admitance(self, force, rot, success):
        ##first think when calling admittance controller should be if current is in vecinity of desired. IF it is, update Xd and run controller. If its not, mantain current Xd and rerun controller. 

        pose = self.robot.get_ee_pose() #TCP T matrix


        force, rot, success = utility._get_contact_info(model=self.m, data=self.d, actor='gripper', obj='pikachu') #FORCE READING

        if success: 
            if not self.aligned:
                self.target_force = np.array([0, 0, 0]) #If in contact no pushing force against object
                #then we align
                
                r = utility.directionToNormal(
                    pose.R,
                    force, 
                    rot=rot
                    )
                #self.align = r.as_matrix()
                
                self.wanted_pose = utility.get_ee_transformation(pose.R, pose.t, r.as_matrix()) 

                #accx = np.linalg.inv(M) @ (wrench[0] + self.target_force - D * velx - K * pos_error[0])
                #velz = self.int_acc(accz, velz, self.dt)
                #Xez = self.int_vel(velz, pos_error[2], self.dt) #Xez = self.int_vel(velz, Xez, self.dt)

                print("Aligned")
                self.aligned = True


            else: #enter here if Aligned = true, mantain pose.R

                pose = self.robot.get_ee_pose() #TCP T matrix
                self.wrench = force #forces in contact frame, wrt base frame. 

                M =  self.M_prev #@ self.align
                K =  self.K_prev #@ self.align
                D =  self.D_prev #@ self.align

            
                # Xd = np.copy(self.Xc) + np.array([0.01, 0.01, 0]) #desired next step
                # Xd = np.copy(self.Xc)

                
                #pos_error = self.Xc - Xd #initially [0,0,0]
                if self.xe_init:
                    self.Xe = self.Xc - Xd
                else: 
                    self.xe_init = True
                
                #print("Type of wrench:", wrench) = 3x1 forces
                #print("Type of target force:", self.target_force) = 3x1 value
                #print("Type of vel:", velx) = 3x1 value
                #print("Type of pos errr:", pos_error) 3x1 values
                #print("Type of K:", K) 3x3
                #print("Type of D:", D) 3x3
                #print("Type of M:", M) 3x3
                

                # Step 1: Calculate acceleration
                self.acc = np.linalg.inv(M) @ (self.wrench + self.target_force - D @ self.vel - K @ self.Xe)

                # Step 2: Integrate acceleration to get velocity
                self.vel = self.int_acc(self.acc, self.vel, self.dt)

                # Step 3: Integrate velocity to get position
                self.Xe = self.int_vel(self.vel, self.Xe, self.dt)
                #print("Position update: ", self.Xe)

                # Step 4: Update compliant position
                self.Xc = self.Xe + Xd
                print("Compliant pos: ", self.Xc)

                self.wanted_pose = SE3.Rt(pose.R, self.Xc)

                self.aligned=False

        else: #if not in contact
            print("Not in contact")
            self.target_force = np.array([0, 0, -2]) #force in Z axis
            self.wrench = np.array([0, 0, 0]) #no forces if not in contact

            pose = self.robot.get_ee_pose() #TCP T matrix
            self.wrench = force #forces in contact frame, wrt base frame. 

            M =  self.M_prev #@ self.align
            K =  self.K_prev #@ self.align
            D =  self.D_prev #@ self.align

        
            # Xd = np.copy(self.Xc) + np.array([0.01, 0.01, 0]) #desired next step
            # Xd = np.copy(self.Xc)
            Xd = self.Xc
            
            #pos_error = self.Xc - Xd #initially [0,0,0]
            if self.xe_init:
                self.Xe = self.Xc - Xd
            else: 
                self.xe_init = True
            
            #print("Type of wrench:", wrench) = 3x1 forces
            #print("Type of target force:", self.target_force) = 3x1 value
            #print("Type of vel:", velx) = 3x1 value
            #print("Type of pos errr:", pos_error) 3x1 values
            #print("Type of K:", K) 3x3
            #print("Type of D:", D) 3x3
            #print("Type of M:", M) 3x3
            

            # Step 1: Calculate acceleration
            self.acc = np.linalg.inv(M) @ (self.wrench + self.target_force - D @ self.vel - K @ self.Xe)

            # Step 2: Integrate acceleration to get velocity
            self.vel = self.int_acc(self.acc, self.vel, self.dt)

            # Step 3: Integrate velocity to get position
            self.Xe = self.int_vel(self.vel, self.Xe, self.dt)
            #print("Position update: ", self.Xe)

            # Step 4: Update compliant position
            self.Xc = self.Xe + Xd
            print("Compliant pos: ", self.Xc)

            self.wanted_pose = SE3.Rt(pose.R, self.Xc)


            #pose = self.robot.get_ee_pose() #TCP T matrix
            #self.wanted_pose = utility.get_ee_transformation(pose.R, self.Xc, pose.R) #self.align

            

            #Construct T matrix to pass onto robot: 
            # T_prev = self.create_T(rot_align.as_matrix(), self.Xc) #BASE FRAME
            #T_prev = self.create_T(self.align, self.Xc) #BASE FRAME


            #else:
                
                #else:
                    #print("Not converging")
                    # Move towards center if not converging
                    #step_start = time.time()
                    #self.wanted_pose = self.robot.move_to_center(self.object_center, step_size=0.02)
                    #print("Moving towards the center")
                    #aligned = False
        
        return self.wanted_pose  #return T to update desired end-effector position
        

    def directionToNormal(self, TCP_R, force):

        """
            Inputs: TCP rotation (3x3 matrix), force direction (3x1 vector XYZ)
            Calulates the direction the robot should turn to align with the surface normal
            Returns: Euler angles for rotation
            If the end effector is parallel to the surface, the rotation matrix should be close to the identity matrix.
        """
        #print(force)
        force_norm = force / np.linalg.norm(force) # Normalize the force vector to be unit
        z_axis = np.atleast_2d([0, 0, 1]) # Axis to align with
        rot = Rotation.align_vectors(z_axis, [force_norm])[0] # Align force to z axis
        return rot.as_matrix() @ TCP_R # New rotation matrix the robot should have to be aligned on base frame. 

    def desired_force():
        Kp = 1 #set by user based on stiffness

        force_value = 4 * Kp #4N to keep contact on object

        return force_value

    def tool_to_base(self, tool_frame):

        """
        Transform a 4x4 T matrix in tool_frame to base frame.
        Returns final T matrix
        """

        T_base_ee = np.array([
        [-0.52960, 0.74368, 0.40801, 0.27667],
        [0.84753, 0.44413, 0.29059, -0.60033],
        [0.03490, 0.49970, -0.86550, 0.51277],
        [0.00000, 0.00000, 0.00000, 1.00000]
        ])

        T_tool_tcp = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.124], #0.124
        [0.00000, 0.00000, 0.00000, 1.00000]])
        
        # Multiply tool_frame by the identity matrix
        final = tool_frame @ T_base_ee @ T_tool_tcp

        #positional_part = final[:3, 3]
        

        # Create an instance of SE3 using SE3.Rt
        return final
    
    def base_to_ee_transform(self, base_T):
        # Convert base_T to Rotation and translation components
        base_rot = Rotation.from_matrix(base_T[:3, :3])
        base_trans = base_T[:3, 3]

        T_base_ee = np.array([
        [-0.52960, 0.74368, 0.40801, 0.27667],
        [0.84753, 0.44413, 0.29059, -0.60033],
        [0.03490, 0.49970, -0.86550, 0.51277],
        [0.00000, 0.00000, 0.00000, 1.00000]
        ])

        # Convert ee_T_base to Rotation and translation components
        ee_rot_base = Rotation.from_matrix(T_base_ee[:3, :3])
        ee_trans_base = T_base_ee[:3, 3]

        # Calculate the rotation and translation components of ee_T
        ee_rot = ee_rot_base * base_rot.inv()  # Rotation relative to base frame
        ee_trans = base_rot.inv().apply(ee_trans_base - base_trans)  # Translation relative to base frame

        # Construct the transformation matrix for EE frame
        ee_T = np.eye(4)
        ee_T[:3, :3] = ee_rot.as_matrix()
        ee_T[:3, 3] = ee_trans

        return ee_T

    
    def create_T(self, R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        T[3, :] = [0, 0, 0, 1]  # Set the last row
        return T

    def euler_to_rotation_matrix(angles):
        """
        Convert Euler angles (in radians) to a rotation matrix using XYZ convention.

        Parameters:
            angles (tuple or list): Euler angles in radians, in the order of (roll, pitch, yaw).

        Returns:
            numpy.ndarray: 3x3 rotation matrix representing the orientation.
        """
        roll, pitch, yaw = angles

        # Calculate sines and cosines of Euler angles
        sr, sp, sy = np.sin(roll), np.sin(pitch), np.sin(yaw)
        cr, cp, cy = np.cos(roll), np.cos(pitch), np.cos(yaw)

        # Define rotation matrices
        Rx = np.array([[1, 0, 0],
                    [0, cr, -sr],
                    [0, sr, cr]])

        Ry = np.array([[cp, 0, sp],
                    [0, 1, 0],
                    [-sp, 0, cp]])

        Rz = np.array([[cy, -sy, 0],
                    [sy, cy, 0],
                    [0, 0, 1]])

        # Combine rotation matrices (order: roll -> pitch -> yaw)
        R = np.dot(Rz, np.dot(Ry, Rx))
        
        return R


    def mat2quat(self, rmat):
        """
        Converts given rotation matrix to quaternion.

        Args:
            rmat (np.array): 3x3 rotation matrix

        Returns:
            np.array: (x,y,z,w) float quaternion angles
        """
        M = np.asarray(rmat).astype(np.float32)[:3, :3]

        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
                [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
                [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is Eigen vector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        inds = np.array([3, 0, 1, 2])
        q1 = V[inds, np.argmax(w)]
        if q1[0] < 0.0:
            np.negative(q1, q1)
        inds = np.array([1, 2, 3, 0])
        return q1[inds]
    

    def quat2mat(self, quaternion):
        """
        Converts given quaternion to matrix.

        Args:
            quaternion (np.array): (x,y,z,w) vec4 float angles

        Returns:
            np.array: 3x3 rotation matrix
        """
        # awkward semantics for use with numba
        inds = np.array([3, 0, 1, 2])
        q = np.asarray(quaternion).copy().astype(np.float32)[inds]

        n = np.dot(q, q)
        if n < EPS:
            return np.identity(3)
        q *= math.sqrt(2.0 / n)
        q2 = np.outer(q, q)
        return np.array(
            [
                [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
                [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
                [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
            ]
        )