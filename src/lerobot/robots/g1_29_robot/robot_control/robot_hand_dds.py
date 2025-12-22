import threading, time, sys
from enum import IntEnum
from multiprocessing import Process, shared_memory, Array, Lock

import numpy as np
import sys
import os

# 👇 动态添加上级目录为模块搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# from ArmHandMotion_Recorder import ArmHandMotionRecorder

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_                               # idl
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_

Dex3_Num_Motors = 7
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"

"""
参考：https://support.unitree.com/home/zh/G1_developer/dexterous_hand
"""
class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6

class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6

class DexControl:
    def __init__(self, idx=0, interfacenet=None, get_state_hz=20, hand_state_callback=None):
        self.hand_state_callback = hand_state_callback
        ChannelFactoryInitialize(idx, interfacenet)
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.RightHandState_subscriber.Init()

        self.subscribe_state_thread = threading.Thread(target=self.subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        self.left_press_info = [{} for _ in range(9)]
        self.right_press_info = [{} for _ in range(9)]

        self.get_state_hz = get_state_hz

        self.max_press = 250000     # 最大压力 30000为无效，正常是100000


    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F  # 4 bits for id
            self.status = status & 0x07  # 3 bits for status
            self.timeout = timeout & 0x01  # 1 bit for timeout

        def _mode_to_uint8(self):
            self.motor_mode |= (self.id & 0x0F)
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode
        

    def control_process(self, left_q_target, right_q_target, dp=0.0, tau=0.0, kp=1.5, kd=0.2, status=0x01):
        dq = dp
        tau = tau
        kp = kp
        kd = kd
        self.left_msg  = unitree_hg_msg_dds__HandCmd_()
        for idx, id in enumerate(Dex3_1_Left_JointIndex):
            ris_mode = self._RIS_Mode(id = id, status = status)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_msg.motor_cmd[id].mode = motor_mode
            self.left_msg.motor_cmd[id].q = left_q_target[idx]
            self.left_msg.motor_cmd[id].dq   = dq
            self.left_msg.motor_cmd[id].tau  = tau
            self.left_msg.motor_cmd[id].kp   = kp
            self.left_msg.motor_cmd[id].kd   = kd

        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for idx, id in enumerate(Dex3_1_Right_JointIndex):
            ris_mode = self._RIS_Mode(id = id, status = status)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_msg.motor_cmd[id].mode = motor_mode  
            self.right_msg.motor_cmd[id].q = right_q_target[idx]
            self.right_msg.motor_cmd[id].dq   = dq
            self.right_msg.motor_cmd[id].tau  = tau
            self.right_msg.motor_cmd[id].kp   = kp
            self.right_msg.motor_cmd[id].kd   = kd  

        if self.check_press_safe(self.left_press_info):
            # print("左手")
            self.LeftHandCmb_publisher.Write(self.left_msg)
        if self.check_press_safe(self.right_press_info):
            # print("右手")
            self.RightHandCmb_publisher.Write(self.right_msg)


    def check_press_safe(self, press_info):
        if press_info[-1] != {}:
            for i in press_info:
                for j in i["pressure"]:
                    if j > self.max_press:
                        print("压力过大")
                        print(j)
                        return False
        return True


    def subscribe_hand_state(self):
        while True:
            left_hand_msg  = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            if left_hand_msg is not None and right_hand_msg is not None:
                for idx, l in enumerate(left_hand_msg.press_sensor_state):
                    press_info = {"pressure": l.pressure, "temperature": l.temperature}
                    self.left_press_info[idx] = press_info
                for idx, r in enumerate(right_hand_msg.press_sensor_state):
                    press_info = {"pressure": r.pressure, "temperature": r.temperature}
                    self.right_press_info[idx] = press_info
            if self.hand_state_callback:
                self.hand_state_callback({"left": self.left_press_info, "right": self.right_press_info})
            time.sleep(1/self.get_state_hz)

    def release(self):
        self.control_process([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], kp=0.0, kd=0.0, status=6)


# if __name__ == "__main__":
#     dc = DexControl(interfacenet="enp129s0")
#     # 获取状态信息
#     # dc.subscribe_hand_state()
#     # 手部指定动作
#     left_target = [-0.03410598263144493, -0.11731448769569397, 1.6460227966308594, -1.0693895816802979, -1.6170012950897217, -1.1778626441955566, -1.4900974035263062]
#     right_target = [-0.03410598263144493, -0.11731448769569397, 1.6460227966308594, -1.0693895816802979, -1.6170012950897217, -1.1778626441955566, -1.4900974035263062]

#     dc.control_process(left_target, right_target)