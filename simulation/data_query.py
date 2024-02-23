from simulation.simulator import MJ 
import numpy as np
from scipy.spatial.transform import Rotation

class Data_Query:
    def __init__(self, rawData):
        self.rawData = rawData

        
    def getFTData(self):
        return self.rawData.sensordata
    

    def directionToNormal(self):
        """
            Calulates the direction the robot should turn to align with the surface normal
            Returns: Euler angles for rotation
        """
        force = self.getFTData()
        force_norm = force / np.linalg.norm(force) # Normalize the force vector to be unit
        y_axis = [0, 1, 0] # Axis to align with
        angle =  np.argcos(np.dot(force_norm, y_axis)) # Rotation angle
        rotation_axis = np.cross(force_norm, y_axis) # Orthorgonal vector (axis to rotate around)
        axis_angle = rotation_axis * angle # Axis angle representation
        return Rotation.from_rotvec(axis_angle).as_euler({'x', 'y', 'z'}, degrees=False) # To euler angles