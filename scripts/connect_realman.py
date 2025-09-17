from Robotic_Arm.rm_robot_interface import *

robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = robot.rm_create_robot_arm("169.254.128.18", 8080, level=0)
print("机械臂ID：", handle.id)

while True:
    print(robot.rm_get_current_arm_state())
