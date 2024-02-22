from simulation.simulator import MJ
from simulation.data_query import Data_Query 
from robot.robot_control import Robot_Controler


if __name__ == "__main__":
  simulator = MJ()
  simulator.start()
  robot_data = Data_Query(simulator.d)
  controller = Robot_Controler(simulator=simulator, sim_data=robot_data)

  while 1:
    print(robot_data.getFTData())
  
    