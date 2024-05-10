#Admittance Controller

import numpy as np
from scipy.spatial.transform import Rotation
from utils import utility
import time
from spatialmath import SE3




class Admitance:
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

        # Gain matrices
        m1 = 1.0
        m2 = 1.0
        m3 = 3.0
        k1 = 1000.0
        k2 = 1000.0
        k3 = 10.0
        kenv1 = 10.0 #set by user depending on current object to be reconstructed
        kenv2 = 10.0
        kd1 = 2*np.sqrt(m1*(k1+kenv1))
        kd2 = 2*np.sqrt(m2*(k2+kenv2))
        kd3 = 2*np.sqrt(m3*(k3))

        self.robot = robot

        self.object_center = [0.4, 0.2, 0.08]

        self.dt = 0.0005 #Based on the control loop of the robot (given by simulation). 1/500 in real UR5 environment. 
        
        self.M_prev = np.array([[m1,0,0],[0,m2,0],[0,0,m3]])
        
        self.K_prev = np.array([[k1,0,0],[0,k2,0],[0,0,0]])  #3 element of 3rd row can be zero        
        
        self.D_prev = np.array([[kd1,0,0],[0,kd2,0],[0,0,kd3]])
        
        self.aligned = False

        self.m = m
        self.d = d
        

        #Initial conditions:
        self.vel = np.array([0, 0, 0])
        self.xe_init = False
        self.Xe = np.array([0, 0, 0])
        self.target_force = np.array([0, 0, 0])

        self.current_TCP = self.robot.get_ee_pose()
        self.Xc = utility._get_pose_from_tran(self.current_TCP)


        self.wanted_pose = None

    def admitance(self, force, rot, success):

        print(self.robot.get_ee_pose())


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
            

            # Step 1: Calculate acceleration
            self.acc = np.linalg.inv(M) @ (self.wrench + self.target_force - D @ self.vel - K @ self.Xe)

            # Step 2: Integrate acceleration to get velocity
            self.vel = self.int_acc(self.acc, self.vel, self.dt)

            # Step 3: Integrate velocity to get position
            self.Xe = self.int_vel(self.vel, self.Xe, self.dt)

            # Step 4: Update compliant position
            self.Xc = self.Xe + Xd

            self.current_TCP = self.robot.get_ee_pose()
            curr_pos = utility._get_pose_from_tran(self.current_TCP)
            print("Current pos: ", curr_pos)
            print("Compliant pos: ", self.Xc)
            print("Position error: ", self.Xe)

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
        
        


    def int_acc(self, acc, vel, dt):
        vel = vel + acc * dt
        '''
        k1 = acc
        k2 = acc
        vel = vel + 0.5 * (k1 + k2) * dt  # Second-order Runge-Kutta (Midpoint) method for single values    
        '''
        return vel

    def int_vel(self, vel, pos, dt):
        pos = pos + vel * dt
        '''for i in range(1, len(vel)):
            pos[i] = pos[i-1] + pos[i-1] * dt  # Euler integration'''
        return pos


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
    def get_inertial(self, inertialValues, COM_data):
        i_matrix = np.zeros((6, 6))
        for i in range(6):
            current_matrix = np.zeros((6,6))
            Ii = np.zeros((3,3))
            Ii[0,0] = inertialValues[i]
            Ii[1,1] = i + 1
            Ii[2,2] = i + 2

            poseVector = np.array([COM_data[i,0], COM_data[i,1], COM_data[i,2]])

            skew_matrix = np.array([
            [0, -poseVector[2], poseVector[1]],
            [poseVector[2], 0, -poseVector[0]],
            [-poseVector[1], poseVector[0], 0]
             ])

            mi_ri = skew_matrix * self.link_masses[i]
            identity_matrix = np.eye(3) * self.link_masses[i]

            current_matrix[:3, :3] = Ii
            current_matrix[:3, 3:] = mi_ri.T
            current_matrix[3:, :3] = mi_ri
            current_matrix[3:, 3:] = identity_matrix

            i_matrix = i_matrix + current_matrix
