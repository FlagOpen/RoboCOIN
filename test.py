from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e, rm_inverse_kinematics_params_t

arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
print('create arm')

ret = arm.rm_create_robot_arm("169.254.128.18", 8080, level=1)

software_info = arm.rm_get_arm_software_info()
if software_info[0] == 0:
    print("\n================== Arm Software Information ==================")
    print("Arm Model: ", software_info[1]['product_version'])
    print("Algorithm Library Version: ", software_info[1]['algorithm_info']['version'])
    print("Control Layer Software Version: ", software_info[1]['ctrl_info']['version'])
    print("Dynamics Version: ", software_info[1]['dynamic_info']['model_version'])
    print("Planning Layer Software Version: ", software_info[1]['plan_info']['version'])
    print("==============================================================\n")

print(arm.rm_movej([0, -130, 90, 0, 0, 0, 0], v=30, r=0, connect=0, block=1))
arm.rm_set_gripper_position(1, True, 3)
joint = arm.rm_get_arm_current_trajectory()['data']
print('joint:', joint)
pose = [-0.1965884119272232, 0.37119612097740173, 0.0755152478814125, 1.5720537900924683, 0.698494017124176, 3.1435928344726562]
_, joint = arm.rm_algo_inverse_kinematics(rm_inverse_kinematics_params_t(q_in=joint, q_pose=pose, flag=1))
print(joint)
print(arm.rm_movej(joint, v=30, r=0, connect=0, block=1))