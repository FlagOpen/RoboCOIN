from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
ret = arm.rm_create_robot_arm('169.254.128.20', 8080)