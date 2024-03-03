from simulation.simulator import MJ
from simulation.data_query import Data_Query 
from robot.robot_control import Robot_Controler
from time import sleep
import roboticstoolbox as rtb
import numpy as np
from spatialmath import SE3
import transformations as tf


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
  simulator.sendJoint(ik.q)
  while 1:
    input("press to continue")
    print(simulator.getState())
    currentTCP = controller.forKin(simulator.getState())
    config = controller.invKin(currentTCP)
    simulator.sendJoint(config.q)
    continue
    target = tf.identity_matrix()
    target[:3, :3] = robot_data.directionToNormal(currentTCP.R, [0, 1, 0]) #currentTCP.R
    transform = currentTCP.t
    target[0, 3] = transform.T[0] 
    target[1, 3] = transform.T[1] + 0.05
    target[2, 3] = transform.T[2] 
    #print(target)
    print(currentTCP.t)
    config = controller.invKin(target)
    simulator.sendJoint(config.q)

    
  
    