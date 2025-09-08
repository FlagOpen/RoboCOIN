#!/bin/bash
# 自动生成的设备绑定配置文件
# 生成时间: 2025-09-08 19:40:00

# 清空现有规则文件
sudo sh -c 'echo "" > /etc/udev/rules.d/serial.rules'
sudo sh -c 'echo "" > /etc/udev/rules.d/fisheye.rules'

# 串口设备绑定规则
sudo sh -c 'echo "ACTION==\"add\", KERNELS==\"1-4.2.2:1.0\", SUBSYSTEMS==\"usb\", MODE:=\"0777\", SYMLINK+=\"ttyUSB81\"" >> /etc/udev/rules.d/serial.rules'

# 鱼眼相机绑定规则

# 重新加载规则并触发
sudo udevadm control --reload-rules && sudo service udev restart && sudo udevadm trigger

echo "设备绑定规则已应用，请重新插拔设备以生效"
