import numpy as np
from BaselineRemoval import BaselineRemoval
import matplotlib.pyplot as plt
import sys, os
import torch
from Lib_CNN import Raman_CNN
from reconstruct_boundary import reconstruct_map

folder = 'Final_Runs/'
arr = []
wv = []
with open(folder + 'Run1.csv', 'r') as new_file:
    for i, line in enumerate(new_file):
        if i>0:
            wv.append((float(line.split(',')[0].strip("\""))))
            new_arr = []
            for j in range(len(line.split(","))):
                if j % 2 == 1:
                    new_arr.append(int(line.split(',')[j].strip("\"").strip('"\n')))

            arr.append(new_arr)
            
arr = np.array(arr).T
wv = np.array(wv)

print(wv.shape)


cutoff = 810

wv_n = wv[wv>cutoff]
wavenumber = 10e7/wv_n
index = wv > cutoff
arr_n = arr[:, index]
print(arr_n.shape)
# print(wv_n, arr_n)

mod_arr = []
imod_arr = []
zha_arr = []
for ind, val in enumerate(arr_n):
    baseObj=BaselineRemoval(arr_n[ind])


    Modpoly_output=baseObj.ModPoly(3)
    mod_arr.append(Modpoly_output)

    Imodpoly_output=baseObj.IModPoly(3)
    imod_arr.append(Imodpoly_output)

    Zhangfit_output=baseObj.ZhangFit()
    zha_arr.append(Zhangfit_output)

mod_arr = np.array(mod_arr)
imod_arr = np.array(imod_arr)
zha_arr = np.array(zha_arr)
print(mod_arr.shape)




# plt.figure()
# plt.plot(wv_n, arr_n[53], label='og')
# plt.plot(wv_n, mod_arr[53], label='mod')
# plt.plot(wv_n, Imodpoly_output, label='imod')
# plt.plot(wv_n, Zhangfit_output, label='zhang')
# plt.legend()
# plt.show()

x_new = np.linspace(np.min(wv_n), np.max(wv_n), 3397)
interp_array = []
interpn_array = []
for ind, val in enumerate(mod_arr):
  interp_array.append(np.interp(x_new, wv_n, val)) #appending on long axis, incorrect
interp_array = np.stack(interp_array, axis=1).T

for i in range(interp_array.shape[0]): #normalization
    interp_array[i] = (interp_array[i]-np.min(interp_array[i]))/(np.max(interp_array[i])-np.min(interp_array[i]))


# plt.figure()
# # for i in range(interp_array.shape[0]):
# plt.plot(x_new, interp_array[5])
# plt.show()
# sys.exit()

if __name__ == "__main__":
    class_score = []
    NUM_FEATURES = 3397
    device = torch.device('cpu')
    model = Raman_CNN(1, 3, NUM_FEATURES)
    model.load_state_dict(torch.load('transfer_deploy_mod.pth'))
    model.to(device)

    for i in range(interp_array.shape[0]):
        new_data = torch.from_numpy(interp_array[i]).float()
        new_data = new_data.to(device)
        model.eval()

        with torch.no_grad():
            output = model((new_data.view(1, 1, NUM_FEATURES)))

        prediction = int(torch.max(output.data, 1)[1].cpu().numpy())
        print(prediction)

        if (prediction == 0):
            print ('ICG')
        if (prediction == 1):
            print ('Cy7.5')
        if (prediction == 2):
            print ('Control')

        class_score.append(prediction)
    
    reconstruct_map(14, 6, 2, 22, class_score)