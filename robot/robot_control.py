from simulation.simulator import MJ
from simulation.data_query import Data_Query
import roboticstoolbox as rtb



class Robot_Controler:
    def __init__(self, simulator: MJ, sim_data: Data_Query, robot:rtb.DHRobot):
        self.simulator = simulator
        self.robot = robot

        


    def sendJoint(self, join_values):
        with self.simulator.jointLock:
          for i in range(0, 6):
            self.simulator.joints[i] = join_values[i]
          self.sendPositions = True

    def forKin(self,q):
        return self.robot.fkine(q)


    def invKin(self, desiredTCP):
          q = self.robot.ikine_LM(desiredTCP)
          return q