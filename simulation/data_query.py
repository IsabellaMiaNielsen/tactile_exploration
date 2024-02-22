from simulation.simulator import MJ 
import mujoco

class Data_Query:
    def __init__(self, rawData):
        self.rawData = rawData
        
        
    def getFTData(self):
        return self.rawData.sensordata