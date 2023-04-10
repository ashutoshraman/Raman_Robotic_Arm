import math
import matplotlib.pyplot as plt
import numpy as np

# measured in cm, fix code



def reconstruct_map(arm_length, arm_length_y, degree, num_iter, class_scores):
    x_pos_arr = []
    y_pos_arr = []

    distance = arm_length * (math.pi/180) * degree * num_iter
    chord = 2* arm_length * math.sin(num_iter*(degree/2)*(math.pi/180))
    print(distance, chord)

    for i in range(num_iter):
        x_pos = arm_length * math.cos(i*degree*(math.pi/180))
        y_pos = arm_length * math.sin(i*degree*(math.pi/180))
        x_pos_arr.append(x_pos)
        y_pos_arr.append(y_pos)

    plt.figure()
    plt.plot(x_pos_arr, y_pos_arr)
    plt.show()


    # print(x_pos_arr, y_pos_arr)

    tumor_map = np.zeros((140, 140))
    for i in range(len(x_pos_arr)):
        tumor_map[round(x_pos_arr[i]*10)-1, round(y_pos_arr[i]*10)] = class_scores[i]


    tumor_map = np.rot90(tumor_map, 1)
    plt.figure()
    plt.imshow(tumor_map)
    plt.show()

def stage_recon(x_size, y_size, class_scores):
    pass

if __name__ == "__main__":
    arm_l_x = 14
    y_axis = 5
    increment = .1
    iters = 400
    class_array = np.ones(400)
    class_array[200:300] = 2
    reconstruct_map(arm_l_x, y_axis, increment, iters, class_array)
