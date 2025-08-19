pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126

pip install -e .

mkdir third_party
cd third_party

git clone https://github.com/agilexrobotics/piper_sdk.git
pip install -e piper_sdk

git clone https://github.com/agilexrobotics/pika_sdk.git
pip install opencv-python
pip install pyrealsense2
pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-22.04/wxpython-4.2.3-cp311-cp311-linux_x86_64.whl
pip install Gooey
pip install --no-deps -e pika_sdk

git clone https://github.com/RealManRobot/RM_API.git
mv RM_API/Python/robotic_arm_package/ rm_api/

cd ..