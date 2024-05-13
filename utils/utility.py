import numpy as np
from scipy.spatial.transform import Rotation
import mujoco as mj
from spatialmath import SE3

def _rotation_matrix_to_align_z_to_direction(direction):
    # Normalize direction vector
    direction /= np.linalg.norm(direction)
    # print("normalized force: ", direction)

    # Calculate axis of rotation
    axis = np.cross([0, 0, 1], direction)
    axis /= np.linalg.norm(axis)

    # Calculate angle of rotation
    angle = np.arccos(np.dot([0, 0, 1], direction))

    # Construct rotation matrix using axis-angle representation
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis
    rotation_matrix = np.array([[t*x*x + c, t*x*y - z*s, t*x*z + y*s],
                                [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
                                [t*x*z - y*s, t*y*z + x*s, t*z*z + c]])

    return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation_matrix

def directionToNormal(TCP_R, force, rot):
    # if force[0] == 0 and force[1] == 0 and force[2] == 0:
    #     print("We are not in contact. Nothing to align to.")
    #     return Rotation.from_matrix(TCP_R)
    # print("Force: ", force)
    z_axis = rot[:, 0]
    # print("Z-axis: ", z_axis)
    rot = Rotation.from_matrix(_rotation_matrix_to_align_z_to_direction(z_axis))
    return rot

def get_ee_transformation(rot, ee_position, desired_rot):
    z_direction = rot[:, -1]
    z_offset = .124
    tool_position = ee_position + z_direction * z_offset 
    rotated_tool_pose = SE3.Rt(desired_rot, tool_position)
    rotated_z = rotated_tool_pose.R[:, -1]
    ee_new_pos = rotated_tool_pose.t - rotated_z * z_offset
    return SE3.Rt(desired_rot, ee_new_pos)

def _get_contact_info(model: mj.MjModel, data: mj.MjData, actor:str, obj:str) -> np.ndarray:
    """
    Function to get the actual force torque values from the simulation. 
    Inputs: Data is the MJData from the simulation, actor is the object that we are touching with i.e. the gripper, obj is the object that we are in contact with i.e. pikachu
    Outputs: Numpy array containing the force and torques
    """
    is_in_contact, cs_i = _is_in_contact(model, data, actor, obj)

    if is_in_contact:
        wrench = _get_cs(model=model, data=data, i=cs_i)
        contact_frame = data.contact[cs_i].frame.reshape((3, 3)).T
        #rot = Rotation.from_matrix(contact_frame)
        #print("contact frame: ", contact_frame)#rot.as_euler("XYZ", degrees=True))
        #print(wrench[:3])
        return contact_frame @ wrench[:3], contact_frame, True
    else:
        return np.zeros(3, dtype=np.float64), np.zeros([3, 3], dtype=np.float64), False

def _obj_in_contact(model: mj.MjModel, cs, obj1: str, obj2: str) -> bool:
    cs_ids = [cs.geom1, cs.geom2]
    obj_ids = [model.geom(obj1 + "_").id, model.geom(obj2 + "_").id]

    if all(elem in cs_ids for elem in obj_ids):
        return True
    else:
        return False

def _is_in_contact(model: mj.MjModel, data: mj.MjData, obj1: str, obj2: str) -> tuple[bool,int]:
    i = 0
    for i in range(data.ncon):
        contact = data.contact[i]
        if _obj_in_contact(model, contact, obj1, obj2):
            return (True, i)
    return (False, i)

def _get_cs(model: mj.MjModel, data: mj.MjData, i: int) -> list[float]:
    c_array = np.zeros(6, dtype=np.float64)
    mj.mj_contactForce(model, data, i, c_array)
    return c_array

def _get_rotation_from_tran(ee_tran):
    matrix = np.array(ee_tran)
    return matrix[:3, :3]

def _get_pose_from_tran(ee_tran):
    matrix = np.array(ee_tran)
    return matrix[:-1, -1].reshape(-1)