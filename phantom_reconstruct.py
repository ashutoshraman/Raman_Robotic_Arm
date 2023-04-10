import numpy as np
from BaselineRemoval import BaselineRemoval
import matplotlib.pyplot as plt
import sys, os
import torch
from Lib_CNN import Raman_CNN
# things to try: 1. full scan with 3 class classifier, 2. full scan with 2 class, 3. line-by-line scan with 3 class, 4. line-by-line with 2 class
# Absolutely retrain on phantom spectra and do 2 class with 345ms int time and darkness iyw
folder = 'Phantom_Training_020623/'

arr = []
wv = []
for num in range(1,31):
    piece_arr = []
    with open(folder+str(num)+'.csv', 'r') as new_file:
        for i, line in enumerate(new_file):
            if i>0:
                new_arr = []
                for j in range(len(line.split(','))):
                    new_arr.append(float(line.split(',')[j]))
                piece_arr.append(new_arr)

    piece_arr = np.array(piece_arr).T[2:, :]
    arr.append(piece_arr)
arr = np.concatenate(arr)


# with open(folder+'test345_34.csv', 'r') as new_file:
#     for i, line in enumerate(new_file):
#         if i>0:
#             new_arr = []
#             for j in range(len(line.split(','))):
#                 new_arr.append(float(line.split(',')[j]))
#             arr.append(new_arr)

# arr = np.array(arr).T[2:, :]


print(arr.shape)
wv = np.linspace(790,920,512)


cutoff = 810
wv_n = wv[wv>cutoff]
index = wv > cutoff
arr_n = arr[:, index]

mod_arr = []

for ind, val in enumerate(arr_n):
    baseObj=BaselineRemoval(arr_n[ind])


    Modpoly_output=baseObj.ModPoly(3)
    mod_arr.append(Modpoly_output)
mod_arr = np.array(mod_arr)


x_new = np.linspace(np.min(wv_n), np.max(wv_n), 3397)
interp_array = []
interpn_array = []
for ind, val in enumerate(mod_arr):
  interp_array.append(np.interp(x_new, wv_n, val)) #appending on long axis, incorrect
interp_array = np.stack(interp_array, axis=1).T

for i in range(interp_array.shape[0]): #normalization
    interp_array[i] = (interp_array[i]-np.min(interp_array[i]))/(np.max(interp_array[i])-np.min(interp_array[i]))


print('now predicting with deployed model')
class_score = []
NUM_FEATURES = 3397
device = torch.device('cpu')
model = Raman_CNN(1, 2, NUM_FEATURES)
model.load_state_dict(torch.load('transfer_deploy_2class_new.pth'))
model.to(device)

for i in range(interp_array.shape[0]):
    new_data = torch.from_numpy(interp_array[i]).float()
    new_data = new_data.to(device)
    model.eval()

    with torch.no_grad():
        output = model((new_data.view(1, 1, NUM_FEATURES)))

    prediction = int(torch.max(output.data, 1)[1].cpu().numpy())
    print(prediction)

    # if (prediction == 0):
    #     print ('ICG')
    #     prediction = 2
    if (prediction == 0):
        print ('Cy7.5')
    if (prediction == 1):
        print ('Control')
    

    class_score.append(prediction)

tumor_map = np.zeros((30,60))
start = 0
end = 60
for ind, row in enumerate(tumor_map):
    if ind % 2 == 0:
        tumor_map[ind] = class_score[start:end]
    else:
        tumor_map[ind] = class_score[end-1:start-1:-1]
    start += 60
    end +=60

tumor_map[tumor_map==1] = 2
tumor_map[tumor_map==0] = 1
tumor_map[tumor_map==2] = 0

plt.figure()
plt.imshow(tumor_map)
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
# plt.legend()
plt.show()

new_count = np.size(tumor_map[tumor_map==0])
print(new_count)


#ideal mask is as follows

def create_circle(rows, cols, center, radius):
    """
    Function to create a filled circle of ones in a numpy array of zeros
    """
    image = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i - center[0])**2 + (j - center[1])**2 <= radius**2:
                image[i, j] = 1
    return image

# Define the size of the array
rows, cols = 30, 60

# Create the array of zeros
image = np.zeros((rows, cols), dtype=np.uint8)

# Define the center coordinates of the two circles
center1 = (13, 27)
center2 = (13, 4.5)

# Define the radius of the two circles
radius1 = 10
radius2 = 2.5

# Create the first circle
circle1 = create_circle(rows, cols, center1, radius1)

# Create the second circle
circle2 = create_circle(rows, cols, center2, radius2)

# Add the two circles to the array of zeros
image = np.add(image, circle1)
image = np.add(image, circle2)



# Display the image
plt.imshow(image)
plt.show()

#calculate IoU

def calculate_iou(gt_mask, pred_mask):
    """
    Calculate Intersection over Union (IoU) between two masks
    """
    # Convert the masks to binary arrays
    gt_mask = gt_mask.astype(np.bool_)
    pred_mask = pred_mask.astype(np.bool_)
    
    # Calculate the intersection between the two masks
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    
    # Calculate the union between the two masks
    union = np.logical_or(gt_mask, pred_mask).sum()
    
    # Calculate the IoU
    iou = intersection / union
    
    return iou

iou = calculate_iou(image, tumor_map)

print("IoU: ", iou)