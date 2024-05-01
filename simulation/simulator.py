import time
from threading import Thread, Lock
import mujoco
import mujoco.viewer
import numpy as np
import roboticstoolbox as rtb
import time
import glfw
from robot.robot_control import Robot
from spatialmath import SE3
from robot.admittance_Controller import Admitance
from utils import utility
from time import sleep

class MJ:
  def __init__(self):
    self.m = mujoco.MjModel.from_xml_path('scene_files/scene.xml')
    self.d = mujoco.MjData(self.m)
    self._data_lock = Lock()
    self.robot = Robot(m=self.m, d=self.d)
    self.angle = 1
    self.step_size = 0.01
    self.object_center = [0.35, 0.2, 0.08]
    self.run_control = False
    
  def run(self) -> None:
    self.th = Thread(target=self.launch_mujoco, daemon=True)
    self.th.daemon = True
    self.th.start()
    input()
    print("done...")
 
  def key_cb(self, key):
    """
    Function for debugging. 
    space should make the robot stay where it is,
    , takes the robot to home position,
    . prints the end effector pose, 
    t touches the top,
    d touches the top directly,
    s touches the side,
    b touches the back
    """
    if key == glfw.KEY_SPACE:
      T_curr = self.robot.get_ee_pose()
      T_des = T_curr @ SE3.Ty(0)
      self.robot.set_ee_pose_compared(T_des)

    if key == glfw.KEY_COMMA:
      self.robot.home()

    if key == glfw.KEY_T:
      self.robot.touch()

    if key == glfw.KEY_D:
      self.robot.direct_touch()

    if key == glfw.KEY_B:
      self.robot.back_touch()

    if key == glfw.KEY_S:
      self.robot.side_touch()

    if key == glfw.KEY_U:
      self.robot.up()

    if key == glfw.KEY_PERIOD:
      print("ee pose = \n", self.robot.get_ee_pose())

    if key ==  glfw.KEY_F:
      print("Force: ", utility._get_contact_info(model=self.m, data=self.d, actor='gripper', obj='pikachu')[0])#self.d.sensordata)

    if key ==  glfw.KEY_C:
      self.robot.move_to_center(self.object_center, step_size=0.05)

    if key == glfw.KEY_A:
      # Align to force
      pose = self.robot.get_ee_pose()
      force, rot, success = utility._get_contact_info(model=self.m, data=self.d, actor='gripper', obj='pikachu')
      if success:
        print("current pose: ", pose)
        
        r = utility.directionToNormal(
          pose.R,
          force, 
          rot=rot
        )

        rotated_pose = utility.get_ee_transformation(pose.R, pose.t, r.as_matrix()) #SE3.Rt(r.as_matrix(), pose.t)

        print("changed pose: ", rotated_pose)
        self.robot.set_ee_pose(rotated_pose)

    if key == glfw.KEY_P:
      self.robot.move_parallel(self.step_size, self.angle)
      self.angle += 1
      self.step_size += 0.002

    if key == glfw.KEY_G:
      if self.run_control:
        self.run_control = False
      else:
        self.run_control = True       
        
    if key == glfw.KEY_UP:
      # Align to force
      pose = self.robot.get_ee_pose()
      print("current pose: ", pose)
      r = utility.directionToNormal(
        pose.R,
        [0, 0, 1]
      )
      rotated_pose = SE3.Rt(r, pose.t)
      print("changed pose: ", rotated_pose)
      self.robot.set_ee_pose(rotated_pose)

    if key == glfw.KEY_LEFT:
      # Align to force
      pose = self.robot.get_ee_pose()
      print("current pose: ", pose)
      r = utility.directionToNormal(
        pose.R,
        [1, 0, 0]
      )
      rotated_pose = SE3.Rt(r, pose.t)
      print("changed pose: ", rotated_pose)
      self.robot.set_ee_pose(rotated_pose)

    if key == glfw.KEY_RIGHT:
      # Align to force
      pose = self.robot.get_ee_pose()
      print("current pose: ", pose)
      r = utility.directionToNormal(
        pose.R,
        force=[0, 1, 0]
        )
      rotated_pose = SE3.Rt(r, pose.t)
      print("changed pose: ", rotated_pose)
      self.robot.set_ee_pose(rotated_pose)

  def launch_mujoco(self):
    aligned = False
    with mujoco.viewer.launch_passive(self.m, self.d, key_callback=self.key_cb) as viewer:
      wrench = utility._get_contact_info(model=self.m, data=self.d, actor='gripper', obj='pikachu')
      controller = Admitance(target_force = 1, wrench=wrench, curent_TCP=self.robot.get_ee_pose(), model_data=self.d, dt=0.002)
      while viewer.is_running():
        # print(self.d.sensordata)
        
        step_start = time.time()
        with self._data_lock:
          mujoco.mj_step(self.m, self.d)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
          sleep(time_until_next_step)
        if self.run_control:
          force, rot, success = utility._get_contact_info(model=self.m, data=self.d, actor='gripper', obj='pikachu')
          if success: # If contact
            if not aligned:
              # Align 
              pose = self.robot.get_ee_pose()
              r = utility.directionToNormal(
                pose.R,
                force, 
                rot=rot
              )
              rotated_pose = utility.get_ee_transformation(pose.R, pose.t, r.as_matrix()) 
              self.robot.set_ee_pose(rotated_pose)
              print("Aligned")
              aligned = True
            else:
              pose = self.robot.get_ee_pose()
              if np.allclose(pose, rotated_pose, rtol=1e-03): # We have aligned correctly
                # Move parallel to the surface
                self.robot.move_parallel(self.step_size, self.angle)
                self.angle += 1
                if self.step_size > 0.05: # Reset step size
                  self.step_size = 0.01
                else:
                  self.step_size += 0.005
                print("Moving along the surface")
                aligned = False
              else:
                self.robot.set_ee_pose(rotated_pose)
                aligned = False # Continue aligning
              
          else:
            # Move towards center if no contact
            step_start = time.time()
            self.robot.move_to_center(self.object_center, step_size=0.05)
            print("Moving towards the center")
            aligned = False
        
