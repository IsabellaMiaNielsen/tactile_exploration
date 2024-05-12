import time
from threading import Thread, Lock
import mujoco
import mujoco.viewer
import numpy as np
import roboticstoolbox as rtb
from spatialmath.base import r2q
import time
import glfw
from robot.robot_control import Robot
from spatialmath import SE3
from robot.admittance_Controller import Admittance
from utils import utility


class MJ:
  def __init__(self):
    self.m = mujoco.MjModel.from_xml_path('scene_files/scene.xml')
    self.d = mujoco.MjData(self.m)
    self._data_lock = Lock()
    self.robot = Robot(m=self.m, d=self.d)
    self.run_control = False
    self.object_center = [0.35, 0.2, 0.08]
    
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

    if key == glfw.KEY_PERIOD: # Dot
      print("ee pose = \n", self.robot.get_ee_pose())

    if key ==  glfw.KEY_F:
      print("Force: ", utility._get_contact_info(model=self.m, data=self.d, actor='gripper', obj='pikachu'))#self.d.sensordata)

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
    #Move robot to TOP touch pose:
    self.robot.home()
    
    # Initialize the Admittance controller outside the loop
    controller = Admittance(self.robot, self.m, self.d)

    with mujoco.viewer.launch_passive(self.m, self.d, key_callback=self.key_cb) as viewer:

      while viewer.is_running():
        step_start = time.time()
        with self._data_lock:
          mujoco.mj_step(self.m, self.d)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # If key "G" was pressed, run controller
        if self.run_control:
          curr_pose_SE3 = self.robot.get_ee_pose()
          curr_quat = r2q(curr_pose_SE3.R, order='xyzs')
          # target = np.array([curr_pose_SE3.t[0] - 0.01, curr_pose_SE3.t[1] + 0.01, curr_pose_SE3.t[2], curr_quat[0], curr_quat[1], curr_quat[2], curr_quat[3]])

          # Always moves in the direction [-0.005, 0.005, 0] in end-effector/TCP frame
          translation_frame = SE3.Rt(np.eye(3), [-0.002, -0.002, 0])
          target_frame = curr_pose_SE3 * translation_frame

          # Construct the target pose
          target = np.concatenate([target_frame.t, curr_quat])
          print(target)
          controller.target = target

          target_reached = False         
          while not target_reached:
            target_reached = controller.step()

            with self._data_lock:
              mujoco.mj_step(self.m, self.d)
            viewer.sync()
      

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
          time.sleep(time_until_next_step)