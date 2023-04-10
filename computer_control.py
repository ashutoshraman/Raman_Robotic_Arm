import pyautogui
import pyfirmata
import time
import sys

# pyautogui.moveTo(2975, 2100)

# sys.exit()

# def moveRaman():
#     base.write(pos)
#     j1.write(256)
#     j2.write(125)
#     j3.write(20)
#     time.sleep(1)

# board = pyfirmata.Arduino('COM10')
# base = board.get_pin('d:9:s')
# j1 = board.get_pin('d:6:s')
# j2 = board.get_pin('d:11:s')
# j3 = board.get_pin('d:10:s')

# pos = 0

#2 class is 5-3
#3 class is 3-4-1 at 0


#need to set to measure mode, 1 scan, laser on high, 1 scan avg, copy as text data, open xl file
# make sure you have cmd prompt, vs code, arduino ide, raman, excel
for i in range(13):
# pyautogui.moveTo(2800,2100)
   
    # moveRaman()
    # pos += 2

    pyautogui.click(2900,2100)
    pyautogui.sleep(3)


    pyautogui.click(950, 200)
    pyautogui.sleep(2)
    pyautogui.click(1625, 200)
    pyautogui.sleep(2)

    pyautogui.click(2975, 2100)
    pyautogui.hotkey('ctrl', 'home')
    # pyautogui.hotkey('ctrl', 'home')
    if i>0:
        for j in range(i):
            pyautogui.press('right')
    pyautogui.hotkey('ctrl', 'v')
    # time.sleep(1)

time.sleep(1000)

# pyautogui.hotkey('win', 'r')
# pyautogui.typewrite('excel')
# pyautogui.hotkey('enter')
# pyautogui.sleep(3)
# pyautogui.hotkey('enter')
# pyautogui.sleep(5)



