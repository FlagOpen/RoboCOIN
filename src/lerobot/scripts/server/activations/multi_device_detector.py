#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多设备自动检测与绑定工具
作用：根据用户输入的设备数量，依次引导用户插入设备并自动检测绑定
输出：生成udev规则文件，实现设备与固定端口的绑定
使用方法：直接运行此脚本 python3 multi_device_detector.py
"""

import subprocess
import re
import os
import cv2
import time
import sys

def run_command(command):
    """
    运行shell命令并返回输出结果
    
    参数:
        command: 要执行的shell命令
        
    返回:
        命令的标准输出内容，如果出错则返回None
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        print(f"执行命令时出错: {str(e)}")
        return None

def get_device_info(device_index):
    """
    获取设备信息，包括RealSense序列号、ttyUSB路径和鱼眼相机路径
    
    参数:
        device_index: 设备序号，用于显示提示信息
        
    返回:
        (serial_number, usb_path, video_path): 设备的序列号、USB路径和视频设备路径
    """
    # 运行 rs-enumerate-devices 命令获取RealSense相机信息
    print(f"\n正在检测第{device_index}个设备的RealSense相机...")
    rs_output = run_command("rs-enumerate-devices -s")
    if not rs_output:
        print("警告: 未检测到RealSense设备，请确保设备已正确连接")
        print("继续尝试检测其他设备信息...")
        serial_number = f"unknown_device_{device_index}"
    else:
        # 解析输出获取序列号
        serial_match = re.search(r'Intel RealSense D405\s+(\d+)', rs_output)
        if not serial_match:
            print("警告: 未找到RealSense D405设备，尝试查找其他型号...")
            # 尝试匹配其他可能的型号
            alt_match = re.search(r'Intel RealSense\s+\w+\s+(\d+)', rs_output)
            if alt_match:
                serial_number = alt_match.group(1)
                print(f"找到其他型号RealSense相机，序列号: {serial_number}")
            else:
                print("警告: 未找到任何RealSense设备序列号")
                serial_number = f"unknown_device_{device_index}"
        else:
            serial_number = serial_match.group(1)
            print(f"找到RealSense D405相机，序列号: {serial_number}")

    # 运行 udevadm 命令获取ttyUSB设备信息
    print(f"\n正在检测第{device_index}个设备的ttyUSB端口...")
    
    # 获取所有ttyUSB设备
    ttyusb_devices = run_command("ls /dev/ttyUSB*").split()
    if not ttyusb_devices:
        print("错误: 未检测到任何ttyUSB设备，请确保设备已正确连接")
        return serial_number, None, None
    
    # 默认使用第一个ttyUSB设备
    ttyusb_device = ttyusb_devices[0]
    print(f"找到ttyUSB设备: {ttyusb_device}")
    
    # 获取设备路径信息
    udev_output = run_command(f"udevadm info {ttyusb_device} | grep DEVPATH")
    if not udev_output:
        print(f"错误: 无法获取{ttyusb_device}的设备路径信息")
        return serial_number, None, None

    # 解析 USB 路径
    usb_path = udev_output[:udev_output.find("ttyUSB")][:-1]  # 获取 1-13.2.4:1.0 这样的格式
    usb_path = usb_path[usb_path.rfind("/")+1:]
    print(f"ttyUSB设备路径: {usb_path}")
    
    print(f"\n正在为第{device_index}个设备寻找鱼眼摄像头...")
    print("请在出现鱼眼摄像头时在弹出的画面窗口中按下s键，非鱼眼摄像头则按下q键")
    print("提示: 鱼眼相机通常会显示圆形或畸变的图像")
    
    video_path = None
    # 遍历可能的视频设备
    for i in range(60):  # 检查前60个视频设备
        try:
            cv2.setLogLevel(0)
            cap = cv2.VideoCapture(i)
            if not cap.isOpened():
                continue
                
            # 设置相机参数
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"正在检测端口: /dev/video{i}")
            print("按s键选择此相机作为鱼眼相机，按q键跳过此相机")
            
            # 显示相机画面，等待用户按键
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 在图像上显示提示信息
                cv2.putText(frame, f"/dev/video{i}", (5, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow(f"device {device_index} - /dev/video{i}", frame)
                key = cv2.waitKey(1)
                
                # 用户按q跳过当前相机，按s选择当前相机
                if key & 0xFF == ord('q'):
                    print(f"跳过相机 /dev/video{i}")
                    break
                elif key & 0xFF == ord('s'):
                    video_path = f'video{i}'
                    print(f"已选择鱼眼相机: /dev/{video_path}")
                    break
                
            cap.release()
            cv2.destroyAllWindows()
            
            # 用户选择了当前相机
            if video_path:
                break
                
        except Exception as e:
            print(f"检测视频设备 {i} 时出错: {str(e)}")
            continue
    
    # 如果没有找到鱼眼相机
    if video_path is None:
        print("警告: 未找到鱼眼相机，或用户未选择任何相机")
        return serial_number, usb_path, None
    
    # 获取视频设备的USB路径
    udev_output = run_command(f"udevadm info /dev/{video_path} | grep DEVPATH")
    if not udev_output:
        print(f"警告: 无法获取 /dev/{video_path} 的设备路径信息")
        return serial_number, usb_path, None
        
    # 解析视频设备的USB路径
    video_usb_path = udev_output[:udev_output.find("video")][:-1]
    video_usb_path = video_usb_path[video_usb_path.rfind("/")+1:]
    print(f"鱼眼相机USB路径: {video_usb_path}")
    
    return serial_number, usb_path, video_usb_path

def generate_setup_bash(devices_info):
    """
    生成设备绑定的udev规则脚本
    
    参数:
        devices_info: 包含所有设备信息的列表，每个元素为(序号, 序列号, USB路径, 视频路径)
        
    返回:
        生成的脚本文件路径
    """
    # 生成文件头
    content = """#!/bin/bash
# 自动生成的设备绑定配置文件
# 生成时间: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """

# 清空现有规则文件
sudo sh -c 'echo "" > /etc/udev/rules.d/serial.rules'
sudo sh -c 'echo "" > /etc/udev/rules.d/fisheye.rules'

"""

    # 为每个设备生成串口规则
    content += "# 串口设备绑定规则\n"
    for device in devices_info:
        index, serial, usb_path, video_path = device
        if not usb_path:  # 跳过无效的USB路径
            continue
            
        tty_index = 80 + index  # 从ttyUSB80开始编号
        content += f'sudo sh -c \'echo "ACTION==\\"add\\", KERNELS==\\"{usb_path}\\", SUBSYSTEMS==\\"usb\\", MODE:=\\"0777\\", SYMLINK+=\\"ttyUSB{tty_index}\\"" >> /etc/udev/rules.d/serial.rules\'\n'
    
    content += "\n# 鱼眼相机绑定规则\n"
    # 为每个设备生成鱼眼相机规则
    for device in devices_info:
        index, serial, usb_path, video_path = device
        if not video_path:  # 跳过无效的视频路径
            continue
            
        video_index = 80 + index  # 从video80开始编号
        content += f'sudo sh -c \'echo "ACTION==\\"add\\", KERNEL==\\"video[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60]*\\",' \
                  f' KERNELS==\\"{video_path}\\", SUBSYSTEMS==\\"usb\\", MODE:=\\"0777\\", SYMLINK+=\\"video{video_index}\\"" >> /etc/udev/rules.d/fisheye.rules\'\n'
    
    # 应用规则
    content += """
# 重新加载规则并触发
sudo udevadm control --reload-rules && sudo service udev restart && sudo udevadm trigger

echo "设备绑定规则已应用，请重新插拔设备以生效"
"""

    # 写入文件
    with open("setup.bash", "w") as f:
        f.write(content)
    os.chmod("setup.bash", 0o755)
    
    return "setup.bash"

def generate_device_info_file(devices_info):
    """
    生成设备信息配置文件，便于其他程序读取
    
    参数:
        devices_info: 包含所有设备信息的列表，每个元素为(序号, 序列号, USB路径, 视频路径)
        
    返回:
        生成的配置文件路径
    """
    content = """# 设备信息配置
# 自动生成时间: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """

"""
    
    for device in devices_info:
        index, serial, usb_path, video_path = device
        tty_index = 80 + index  # 从ttyUSB80开始编号
        video_index = 80 + index  # 从video80开始编号
        
        content += f"# 设备 {index}\n"
        content += f"DEVICE_{index}_SERIAL={serial}\n"
        
        if usb_path:
            content += f"DEVICE_{index}_TTY_PATH=/dev/ttyUSB{tty_index}\n"
        else:
            content += f"DEVICE_{index}_TTY_PATH=未检测到\n"
            
        if video_path:
            content += f"DEVICE_{index}_VIDEO_PATH=/dev/video{video_index}\n"
        else:
            content += f"DEVICE_{index}_VIDEO_PATH=未检测到\n"
            
        content += "\n"
    
    # 写入文件
    with open("devices_info.conf", "w") as f:
        f.write(content)
    
    return "devices_info.conf"

def print_colored(text, color="green"):
    """
    打印彩色文本，提升用户体验
    
    参数:
        text: 要打印的文本
        color: 颜色，支持green、yellow、red、blue
    """
    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "blue": "\033[94m",
        "bold": "\033[1m",
        "end": "\033[0m"
    }
    
    print(f"{colors.get(color, '')}{text}{colors['end']}")

def main():
    """
    主函数，执行设备检测和配置流程
    
    流程:
    1. 获取用户输入的设备数量
    2. 依次引导用户插入每个设备并获取信息
    3. 生成设备绑定规则和配置文件
    4. 应用规则并提供使用说明
    """
    print_colored("====================================", "bold")
    print_colored("     多设备自动检测与绑定工具 v1.1   ", "bold")
    print_colored("====================================", "bold")
    
    # 检查必要的工具
    missing_tools = []
    if not run_command("which v4l2-ctl"):
        missing_tools.append("v4l-utils")
    if not run_command("which rs-enumerate-devices"):
        missing_tools.append("realsense-sdk")
    
    if missing_tools:
        print_colored("\n警告: 检测到缺少以下工具，可能影响检测结果:", "yellow")
        for tool in missing_tools:
            print(f"  - {tool}")
        print("\n建议安装:")
        if "v4l-utils" in missing_tools:
            print("  sudo apt install v4l-utils")
        if "realsense-sdk" in missing_tools:
            print("realsense sdk")
        
        print_colored("\n是否继续执行？(y/n)", "yellow")
        choice = input().strip().lower()
        if choice != 'y':
            print_colored("操作已取消", "red")
            return
    
    # 获取用户输入的设备数量
    while True:
        try:
            print_colored("\n请输入需要配置的设备数量，如：", "blue")
            print_colored("\n* 1 个 sense 或 1 个 gripper    数量为： 1 ", "green")
            print_colored("\n* 2 个 sense                   数量为： 2 ", "green")
            print_colored("\n* 2 个 gripper                 数量为： 2 ", "green")
            print_colored("\n* 1 个 sense 和 1 个 gripper    数量为： 2 ", "green")
            print_colored("\n* 2 个 sense 和 1 个 gripper    数量为： 3 ", "green")
            print_colored("\n* 2 个 sense 和 2 个 gripper    数量为： 4 ", "green")
            print_colored("\n请输入需要配置的设备数量，然后按回车键继续...", "blue")
            num_devices = int(input().strip())
            if num_devices <= 0:
                print_colored("设备数量必须大于0，请重新输入", "red")
                continue
            break
        except ValueError:
            print_colored("请输入有效的数字", "red")
    
    devices_info = []
    
    # 依次获取每个设备的信息
    for i in range(1, num_devices + 1):
        print_colored(f"\n===== 配置第 {i}/{num_devices} 个设备 =====", "bold")
        print_colored(f"\n请插入第{i}个设备，然后按回车键继续...", "blue")
        input()
        print_colored(f"正在获取第{i}个设备信息...", "green")
        
        # 获取设备信息
        serial_number, usb_path, video_path = get_device_info(i)
        
        if not serial_number and not usb_path and not video_path:
            print_colored(f"错误: 无法获取第{i}个设备的任何信息", "red")
            print_colored("是否跳过此设备并继续配置下一个设备？(y/n)", "yellow")
            choice = input().strip().lower()
            if choice != 'y':
                print_colored("操作已取消", "red")
                return
            continue
        
        # 保存设备信息
        devices_info.append((i, serial_number, usb_path, video_path))
        
        print_colored(f"\n第{i}个设备信息获取成功:", "green")
        print(f"  序列号: {serial_number}")
        print(f"  USB路径: {usb_path if usb_path else '未检测到'}")
        print(f"  视频路径: {video_path if video_path else '未检测到'}")
        
        if i < num_devices:
            print_colored(f"\n请拔出第{i}个设备，准备插入下一个设备", "yellow")
            print_colored("注意: 请不要插入同一个USB口，配置完成后USB口不能改变", "yellow")
    
    if not devices_info:
        print_colored("\n未获取到任何设备信息，配置失败", "red")
        return
    
    # 生成配置文件
    print_colored("\n正在生成配置文件...", "green")
    setup_file = generate_setup_bash(devices_info)
    info_file = generate_device_info_file(devices_info)
    
    print_colored("\n配置完成！已生成以下文件：", "green")
    print(f"1. {setup_file} - 设备绑定规则脚本")
    print(f"2. {info_file} - 设备信息配置文件")
    
    # 询问是否立即应用规则
    print_colored("\n是否立即应用设备绑定规则？(y/n)", "blue")
    choice = input().strip().lower()
    if choice == 'y':
        print_colored("\n执行设备绑定规则脚本...", "green")
        result = run_command(f"bash {setup_file}")
        print(result if result else "规则应用完成")
        
        print_colored("\n执行完成！", "green")
        print_colored("请拔插所有设备，注意插入先前绑定的同一个USB口", "yellow")
        print(f"设备绑定信息已保存到 {info_file}")
        
        print_colored("\n设备绑定后的使用方法:", "bold")
        print("1. 每个设备都有固定的ttyUSB和video设备路径")
        for device in devices_info:
            index, serial, usb_path, video_path = device
            tty_index = 80 + index
            video_index = 80 + index
            print(f"   设备 {index}: ")
            print(f"     *串口端口号                /dev/ttyUSB{tty_index}")
            print(f"     *鱼眼相机索引              video{video_index}")
            print(f"     *双目深度相机序列号        {serial}")
            
        
        print("\n2. 在代码中使用这些固定路径:")
        print("   from pika import sense")
        print("   my_device = sense('/dev/ttyUSB81')  # 使用第1个设备")
        print("   my_device.set_fisheye_camera_index(81)  # 使用第1个设备的鱼眼相机")
        print("   my_device.set_realsense_serial_number(230322270988)  # 使用序列号为230322270988的Realsense设备")
    else:
        print_colored(f"\n您可以稍后手动执行 bash {setup_file} 应用设备绑定规则", "blue")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_colored("\n\n操作已取消", "red")
        sys.exit(0)
    except Exception as e:
        print_colored(f"\n\n程序执行出错: {str(e)}", "red")
        sys.exit(1)
