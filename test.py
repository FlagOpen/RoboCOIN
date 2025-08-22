#!/usr/bin/env python3

import rospy
import moveit_commander
from sensor_msgs.msg import JointState


def callback(data):
    rospy.loginfo("Received joint states: %s", data.position)


def test_rospy():
    rospy.init_node('test_node', anonymous=True)
    rospy.loginfo("ROS node initialized successfully.")

    # get joint states
    # sub = rospy.Subscriber('/joint_states', JointState, callback)
    # rospy.sleep(1)  # wait for messages to be received
    pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
    rospy.sleep(1)  # wait for the publisher to be set up
    msg = JointState()
    msg.name = [f'joint{i + 1}' for i in range(7)]
    msg.position = [1.0, 0.4, -0.6, 0.0, 0.9, 0.0, 0.0]
    msg.header.stamp = rospy.Time.now()
    rospy.loginfo("Publishing joint states: %s", msg.position)
    pub.publish(msg)
    rospy.sleep(1)  # wait for the message to be sent

def test_moveit():
    # 初始化 MoveIt! 相关组件
    moveit_commander.roscpp_initialize([])
    move_group = moveit_commander.MoveGroupCommander("arm")  # 可以根据需要修改为 "arm"、"gripper"或"piper"

    # 获取当前关节值
    joint_goal = move_group.get_current_joint_values()
    joint_goal[0] = 0.0  # arm使用joint_goal长度为6,gripper使用joint_goal长度为1,piper使用joint_goal长度为7
    # joint_goal[1] = 0.0
    # joint_goal[2] = 0.0
    # joint_goal[3] = 0.0
    # joint_goal[4] = 0.0
    # joint_goal[5] = 0.0
    # joint_goal[6] = 0.0 # 使用piper规划组是，这一位是夹爪控制

    # 设置并执行目标
    move_group.set_joint_value_target(joint_goal)
    success = move_group.go()
    moveit_commander.roscpp_shutdown()

if __name__ == "__main__":
    test_rospy()