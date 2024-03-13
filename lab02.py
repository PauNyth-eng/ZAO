import cv2
import numpy as np
from PIL import ImageGrab
from pynput.mouse import Button, Controller
import time

def template_matching(template, screenshot):
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_loc

def click(x, y):
    mouse = Controller()
    mouse.position = (x, y)
    mouse.click(Button.left)

def simple_capture(template_path):
    template = cv2.imread(template_path)
    template_height, template_width, _ = template.shape

    while True:
        screenshot = np.array(ImageGrab.grab(bbox=(0, 0, 1920, 1080))) 
        screenshot_height, screenshot_width, _ = screenshot.shape
        print('Hello')
        location = template_matching(template, screenshot)

        if location[0] != -1:
            print("Template found at:", location)
            click(location[0] + template_width // 2, location[1] + template_height // 2)
        else:
            print("Template not found")

if __name__ == "__main__":
    template_path = "template.png" # Path to your template image
    simple_capture(template_path)
