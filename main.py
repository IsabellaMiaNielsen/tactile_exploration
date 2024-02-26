from simulation.simulator import MJ
from simulation.data_query import Data_Query 
from robot.robot_control import Robot_Controler
from time import sleep
import roboticstoolbox as rtb
import numpy as np


if __name__ == "__main__":
  robot = rtb.DHRobot(
      [ 
          rtb.RevoluteDH(d=0.1625, alpha = np.pi/2),
          rtb.RevoluteDH(a=-0.425),
          rtb.RevoluteDH(a=-0.3922),
          rtb.RevoluteDH(d=0.1333, alpha=np.pi/2),
          rtb.RevoluteDH(d=0.0997, alpha=-np.pi/2),
          rtb.RevoluteDH(d=0.0996)
      ], name="UR5e"
                  )

  simulator = MJ(robot)
    # Universal Robot UR5e kiematics parameters 


  simulator.start()
  robot_data = Data_Query(simulator.d)
  controller = Robot_Controler(simulator=simulator, sim_data=robot_data,robot=simulator.robot)
  fk = controller.forKin([-0.3, 0, -2.2, 0, 2, 0.7854])
  ik = controller.invKin(fk)

  while 1:
  
    controller.sendJoint(ik.q)
    print(ik)
    
  
    