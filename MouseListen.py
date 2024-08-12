import pynput
from pynput.mouse import Listener


def mouse_click(x, y, button, pressed):
    print(button, pressed)
    if pressed and button == pynput.mouse.Button.x2:
        pass


with Listener(on_click=mouse_click) as Listener:
   Listener.join()