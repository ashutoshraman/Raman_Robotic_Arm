import Lib_DataProcess
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import pyautogui



tid = Lib_DataProcess.ControlStage()
tid.move_to(3, 3)



def create_step_positions(x_start, y_start, x_size, y_size, step_size):
    x_steps = np.linspace(x_start, x_start+x_size, round(x_size/step_size))
    y_steps = np.linspace(y_start, y_start+y_size, round(y_size/step_size))
    return x_steps, y_steps





phrase= 'n' #type anything other than n to break while loop
while phrase == 'n':
    x_pos = int(input('x position?'))
    y_pos = int(input('y position?'))
    tid.move_to(x_pos, y_pos)
    phrase = input('ready to begin scan?')

x_steps, y_steps = create_step_positions(x_pos, y_pos, 30, 60, 1) 
counter = 0

#new stuff
pyautogui.click(2900,2100)
pyautogui.sleep(3)

pyautogui.click(950, 200) #10 45, then 11, 18, laser is .345 and stage is .34 for 30x50 #350 for sensor and 340 for stage
yj= y_pos
for i, xi in enumerate(x_steps):

    tid.move_to(xi, yj)
    if i >0:
        input("ready for next scan?")
        pyautogui.click(2900,2100)
        pyautogui.sleep(3)

        pyautogui.click(950, 200)

    if i % 2 == 0:
        for j, yj in enumerate(y_steps):
            tid.move_to(xi, yj)
            time.sleep(.34)
    else:
        for j, yj in enumerate(y_steps[::-1]):
            tid.move_to(xi, yj)
            time.sleep(.34)
    



sys.exit()


for i, xi in enumerate(x_steps):    
    for j, yj in enumerate(y_steps):

        tid.move_to(xi, yj)
        pyautogui.click(2900,2100)
        pyautogui.sleep(3)


        pyautogui.click(950, 200)
        pyautogui.sleep(.5)
        pyautogui.click(1625, 200)
        # pyautogui.sleep(2)

        pyautogui.click(2975, 2100)
        pyautogui.hotkey('ctrl', 'home')
        # pyautogui.hotkey('ctrl', 'home')
        if counter>0:
            for k in range(counter):
                pyautogui.press('right')
        pyautogui.hotkey('ctrl', 'v')
        counter +=1

    
    



time.sleep(5)
tid.move_to(10,10)
