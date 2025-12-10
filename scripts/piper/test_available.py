import time
from piper_sdk import *


piper = C_PiperInterface_V2("can0")
piper.ConnectPort()

while not piper.EnablePiper():
    print("Waiting for Piper to enable...")
    time.sleep(0.01)

print("Piper enabled!")