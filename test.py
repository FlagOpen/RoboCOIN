# import keyboard
import sys
import time
import threading
from sshkeyboard import listen_keyboard


# def on_press(key):
#     print('press:', key)

# # start a new thread to listen to keyboard events
# listener_thread = threading.Thread(target=listen_keyboard, args=(on_press,))
# listener_thread.start()


class KeyboardListener:
    def __init__(self):
        self._key = None
        self._listener = threading.Thread(target=listen_keyboard, args=(self._on_press,))
        self._listener.daemon = True
        self._listener.start()
    
    def _on_press(self, key):
        self._key = key


listener = KeyboardListener()

while True:
    time.sleep(1)
    print(f"Running... {listener._key}")
    if listener._key == 'q':
        print("Quit signal received.")
        sys.stdin.flush()
        ret = input("Are you sure to quit? (y/n): ")
        print(ret)
        break