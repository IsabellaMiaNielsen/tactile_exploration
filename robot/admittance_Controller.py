#Admittance Controller

import numpy as np
from scipy.spatial.transform import Rotation
from robot.robot_control import Robot
from utils import utility

class Admitance:
    def __init__(self, target_force, wrench, curent_TCP, model_data, dt):

        '''
        INPUTS:
            target_force -> force to push against object (around 4N)
            wrench -> the forces exerted by the surface on the end effector due to contact. Sensor reading (3 element vector, [Fx; Fy; Fz])
            des_vel -> desired velocity during movement. VALUE???
            rot_align -> rotation matrix the robot should have to be aligned with surface normal
            Xc -> current_pos 
            current_vel
        
        RETUNS:
            Xc = desired end effector position based on controller response to external force (surface normal) in TOOL FRAME
        '''
        self.link_masses = [3.761, 8.058, 2.846, 1.37, 1.3, 0.365]
        self.model_data = model_data
        self.wrench = wrench
        self.dt = dt
        self.current_TCP = curent_TCP
        self.m1 = 1
        m2 = 1
        m3 = 3
        k1 = 1
        k2 = 1
        k3 = 3
        kenv1 = 10 #set by user depending on current object to be reconstructed
        kenv2 = 10
        kd1 = 2*np.sqrt(self.m1*(k1+kenv1))
        kd2 = 2*np.sqrt(m2*(k2+kenv2))
        dt = 0.002 #Based on the control loop of the robot (given by simulation). 1/500 in real UR5 environment. 
        
        self.M_prev = np.array([[self.m1,0,0],[0,m2,0],[0,0,m3]])
        
        self.K_prev = np.array([[k1,0,0],[0,k2,0],[0,0,0]])  #3 element of 3rd row can be zero        
        
        self.D_prev = np.array([[kd1,0,0],[0,kd2,0],[0,0,k3]])

        #Initial conditions:
        self.velx = 0
        self.vely = 0
        self.velz = 0
        self.Xc = utility._get_pose_from_tran(self.current_TCP)

        self.first_iteration = True
        
        self.Xex = 0
        self.Xey = 0
        self.Xez = 0
        self.wrench = [0,0,0]

        # Controller Loop
        
        self.probe_in_contact = False
        self.target_force = target_force

    def admitance(self, pobe_in_contact, wrench, current_TCP, model_data):
        if pobe_in_contact :  # You need some condition to break the loop

            self.wrench = wrench #only forces
            self.TCP_R = utility._get_rotation_from_tran(current_TCP)
            print(wrench)

            rot_align = (self.directionToNormal(self.TCP_R, wrench))

            M = rot_align @ self.M_prev #update gains based on orientation function
            K = rot_align @ self.K_prev
            D = rot_align @ self.D_prev

            # M = M_prev #update gains based on orientation function
            # K = K_prev
            # D = D_prev
            
            # Step 1: Calculate acceleration
            Xd = np.copy(self.Xc) + np.array([0.01, 0.01, 0])

            if self.first_iteration:
                Xd = self.Xc
            
            pos_error = self.Xc - Xd
            
            print("Type of wrench:", wrench)
            print("Type of target force:", self.target_force)
            print("Type of vel:", velx)
            print("Type of pos errr:", pos_error)
            print("Type of K:", K)
            print("Type of D:", D)
            print("Type of M:", M)

            accx = np.linalg.inv(self.m1) @ (wrench[0] + self.target_force - D @ velx - K @ pos_error[0])
            accy = np.linalg.inv(self.m1) @ (wrench[1] + self.target_force - D @ vely - K @ pos_error[1])
            accz = np.linalg.inv(self.m1) @ (wrench[2] + self.target_force - D @ velz - K @ pos_error[2])
            
            # Step 2: Integrate acceleration to get velocity
            velx = self.int_acc(accx, velx, self.dt)
            vely = self.int_acc(accy, vely, self.dt)
            velz = self.int_acc(accz, velz, self.dt)
            
            # Step 3: Integrate velocity to get position
            Xex = self.int_vel(velx, Xex, self.dt)
            Xey = self.int_vel(vely, Xey, self.dt)
            Xez = self.int_vel(velz, Xez, self.dt)
            
            # Step 4: Update current position
            Xcx = Xex + Xd[0]
            Xcy = Xez + Xd[1]
            Xcz = Xey + Xd[2]
            self.Xc = [Xcx, Xcy, Xcz]
            
            self.first_iteration = False
            # Exit condition in case force readings are lower than a threshold (contact lost)
            # if wrench >= [0,0,0]:
            #     break
        return self.tool_to_base(self.Xc)  #update desired end-effector position (internal state)


    def int_acc(acc, vel, dt):
        vel = vel + acc * dt
        '''
        k1 = acc
        k2 = acc
        vel = vel + 0.5 * (k1 + k2) * dt  # Second-order Runge-Kutta (Midpoint) method for single values    
        '''
        return vel

    def int_vel(vel, pos, dt):
        pos = pos + vel * dt
        '''for i in range(1, len(vel)):
            pos[i] = pos[i-1] + pos[i-1] * dt  # Euler integration'''
        return pos

    def desired_force():
        Kp = 1 #set by user based on stiffness

        force_value = 4 * Kp #4N to keep contact on object

        return force_value

    def tool_to_base(tool_frame):

        """
        Transform a 4x4 T matrix in tool_frame to base frame.
        Returns only the positional part
        """

        T_base_tool = np.array([
        [-0.52960, 0.74368, 0.40801, 0.27667],
        [0.84753, 0.44413, 0.29059, -0.60033],
        [0.03490, 0.49970, -0.86550, 0.51277],
        [0.00000, 0.00000, 0.00000, 1.00000]
        ])

        T_tool_tcp = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0.1143],
        [0.00000, 0.00000, 0.00000, 1.00000]])
        
        # Multiply tool_frame by the identity matrix
        final = tool_frame @ T_base_tool @ T_tool_tcp

        positional_part = final[:3, 3]

        return positional_part

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