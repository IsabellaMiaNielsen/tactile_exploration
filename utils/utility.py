import numpy as np
from scipy.spatial.transform import Rotation
import mujoco as mj

def directionToNormal(TCP_R, force):
    """
        Inputs: TCP rotation, force direction
        Calulates the direction the robot should turn to align with the surface normal
        Returns: Euler angles for rotation
        If the end effector is parallel to the surface, the rotation matrix should be close to the identity matrix.
    """
    if force[0] == 0 and force[1] == 0 and force[2] == 0:
        print("We are not in contact. Nothing to align to.")
        return TCP_R
    force = [int(np.abs(force[0])), int(np.abs(force[1])), int(np.abs(force[2]))]
    force_norm = force / np.linalg.norm(force) # Normalize the force vector to be unit
    z_axis = np.atleast_2d([1, 0, 0]) # Axis to align with
    rot = Rotation.align_vectors(z_axis, [force_norm])[0] # Align force to z axis
    return rot.as_matrix() @ Rotation.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).as_matrix() # New rotation matrix the robot should have to be aligned. 


def _get_contact_info(model: mj.MjModel, data: mj.MjData, actor:str, obj:str) -> np.ndarray:
    """
    Function to get the actual force torque values from the simulation. 
    Inputs: Data is the MJData from the simulation, actor is the object that we are touching with i.e. the gripper, obj is the object that we are in contact with i.e. pikachu
    Outputs: Numpy array containing the force and torques
    """
    is_in_contact, cs_i = _is_in_contact(model, data, actor, obj)

    if is_in_contact:
        wrench = _get_cs(model=model, data=data, i=cs_i)
        return wrench
    else:
        return np.zeros(6, dtype=np.float64)

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