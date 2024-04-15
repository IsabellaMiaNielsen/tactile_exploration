#Admittance Controller

import numpy as np
from scipy.spatial.transform import Rotation
from robot.robot_control import Robot


def admittance(target_force, robotTCP, sensordata):

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

    m1 = 1
    m2 = 1
    m3 = 3
    k1 = 1
    k2 = 1
    k3 = 3
    kenv1 = 10 #set by user depending on current object to be reconstructed
    kenv2 = 10
    kd1 = 2*np.sqrt(m1*(k1+kenv1))
    kd2 = 2*np.sqrt(m2*(k2+kenv2))
    dt = 0.002 #Based on the control loop of the robot (given by simulation). 1/500 in real UR5 environment. 

    M_prev = np.array([[m1,0,0],[0,m2,0],[0,0,m3]])
    

    K_prev = np.array([[k1,0,0],[0,k2,0],[0,0,0]])  #3 element of 3rd row can be zero
    
    
    D_prev = np.array([[kd1,0,0],[0,kd2,0],[0,0,k3]])

    

    #Initial conditions:
    vel = [0,0,0]
    Xc = robot.get_pose()


    first_iteration = True
    
    Xe = [0,0,0]
    wrench = [0,0,0]
    # Controller Loop
    i = 0
    while i < 1000:  # You need some condition to break the loop

        wrench = robot._d.sensordata[:3] #only forces
        TCP_R = robot.get_rotation()
        print(wrench)

        rot_align = (directionToNormal(TCP_R, wrench))

        M = rot_align @ M_prev #update gains based on orientation function
        K = rot_align @ K_prev
        D = rot_align @ D_prev

        
        # M = M_prev #update gains based on orientation function
        # K = K_prev
        # D = D_prev
        
        # Step 1: Calculate acceleration
        print(Xc)
        Xd = np.copy(Xc) + np.array([0.01, 0.01, 0])

        if first_iteration:
            Xd = Xc
        
        pos_error = Xc - Xd
        
        print("Type of wrench:", wrench)
        print("Type of target force:", target_force)
        print("Type of vel:", vel)
        print("Type of pos errr:", pos_error)
        print("Type of K:", K)
        print("Type of D:", D)
        print("Type of M:", M)

        acc = np.linalg.inv(M) @ (wrench + target_force - D @ vel - K @ pos_error)
        
        # Step 2: Integrate acceleration to get velocity
        vel = int_acc(acc, vel, dt)
        
        # Step 3: Integrate velocity to get position
        Xe = int_vel(vel, Xe, dt)
        
        # Step 4: Update current position
        Xc = Xe + Xd
        
        first_iteration = False
        # Exit condition in case force readings are lower than a threshold (contact lost)
        # if wrench >= [0,0,0]:
        #     break
    
        Xc = Xe + Xd
        i +=1
    return tool_to_base(Xc)


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


def directionToNormal(TCP_R, force):

    """
        Inputs: TCP rotation (3x3 matrix), force direction (3x1 vector XYZ)
        Calulates the direction the robot should turn to align with the surface normal
        Returns: Euler angles for rotation
        If the end effector is parallel to the surface, the rotation matrix should be close to the identity matrix.
    """
    force_norm = force / np.linalg.norm(force) # Normalize the force vector to be unit
    z_axis = np.atleast_2d([0, 0, 1]) # Axis to align with
    rot = Rotation.align_vectors(z_axis, [force_norm])[0] # Align force to z axis
    return rot.as_matrix() @ TCP_R # New rotation matrix the robot should have to be aligned.

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

if __name__ == "__main__":

    print('...')
