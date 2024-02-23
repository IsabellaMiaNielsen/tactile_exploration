from simulation.simulator import MJ
from simulation.data_query import Data_Query



class Robot_Controler:
    def __init__(self, simulator: MJ, sim_data: Data_Query):
        self.simulator = simulator
        pass


    def sendJoint(self,join_values):
        with self.simulator.jointLock:
          for i in range(0, 6):
            self.simulator.joints[i] = join_values[i]
          self.sendPositions = True