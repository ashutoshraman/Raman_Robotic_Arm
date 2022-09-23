import pandas as pd
import numpy as np
import glob
import h5py
import os
import matplotlib.pyplot as plt
import time
import Lib_CNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import KFold
import seaborn as sns
import sys

folder = 'Reference_Raman_Spectra/'
plt.figure()
for i in os.listdir(folder):
    new_spectra = pd.read_csv(folder+i, header=None)
    
    plt.plot(new_spectra[0], new_spectra[1])
    plt.title(i)
    plt.show()
print(new_spectra.shape)
sys.exit()


x1_array, y1_array = H5_Class().read_raster_contents('Mouse_study_day_1.hdf5', 'Data/tumorID-csv/81721-Mouse/healthy/', 0)
x2_array, y2_array = H5_Class().read_raster_contents('Mouse_study_day_1.hdf5', 'Data/tumorID-csv/81721-Mouse/tumor/', 1)
x3_array, y3_array = H5_Class().read_raster_contents('Mouse_study_day_2.hdf5', 'Data/tumorID-csv/083021-mouse/healthy/', 0)
x4_array, y4_array = H5_Class().read_raster_contents('Mouse_study_day_2.hdf5', 'Data/tumorID-csv/083021-mouse/tumor1/', 1)
x5_array, y5_array = H5_Class().read_raster_contents('Mouse_study_day_2.hdf5', 'Data/tumorID-csv/083021-mouse/tumor2/', 1) #could try shuffling within each class before train test split

X_array = np.concatenate((np.transpose(x1_array[:, 1:]), np.transpose(x2_array[:, 1:]), np.transpose(x3_array[:, 1:]), np.transpose(x4_array[:, 1:]), np.transpose(x5_array[:, 1:])), axis=0)
Y_array = np.concatenate((np.transpose(y1_array[:, 1:]), np.transpose(y2_array[:, 1:]), np.transpose(y3_array[:, 1:]), np.transpose(y4_array[:, 1:]), np.transpose(y5_array[:, 1:])), axis=0)
print(X_array.shape, Y_array.shape)

wavelength_500 = np.argmin(np.abs(x1_array[:, 0] - 500))

def data_cleaning(X_data, Y_data): #use only past 405 peak data, convolve to smooth, delete spectra w avg intensity <.02, normalize
    cutoff_405 = np.argmin(np.abs(x1_array[:, 0] - 445)) #this is already done above, outside this function
    wavelength_uppercut = np.argmin(np.abs(x1_array[:, 0] - 750))
    X_data = X_data[:, cutoff_405:wavelength_uppercut] # use only past 405 peak
    for i in range(X_data.shape[0]-1, -1, -1):
        X_data[i, :] = np.convolve(X_data[i, :], np.ones(10)/10, 'same') #convolve to smooth  
    indices = np.where(np.average(X_data, axis=1) >= .005) 
    X_data, Y_data = X_data[indices], Y_data[indices] #only use those with avg apectra >=.02
    X_data = X_data / X_data[:, (wavelength_500 - cutoff_405)].reshape(X_data.shape[0], 1) #normalize with 500nm wvlgth
    # wavelength_max = np.amax(X_data, axis=1)
    # X_data = X_data / wavelength_max.reshape(X_data.shape[0], 1) 

    
    return X_data, Y_data

X_array, Y_array = data_cleaning(X_array, Y_array)
print('shape after cleaning')
print(X_array.shape, Y_array.shape)


# Split into train+val and test, stratify ensures classes have equal representation in train, val, test, since they are disproportionate
X_trainval, X_test, y_trainval, y_test = train_test_split(X_array, Y_array.reshape(Y_array.shape[0]), test_size=0.2, stratify=Y_array.reshape(Y_array.shape[0]), shuffle=True, random_state=69)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, shuffle=True, random_state=21)
# kfold = KFold(10, True, 1)

# Normalize output by (x-min)/(max-min). Fit to train data, and transform val and test using train metrics to prevent leakage
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)

train_dataset = Lib_ANN.DataAccess(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
val_dataset = Lib_ANN.DataAccess(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
test_dataset = Lib_ANN.DataAccess(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())



#Weighted Random Sampler and Weight Distribution
def get_class_distribution(obj):
    count_dict = {
        "class_1": 0,
        "class_2": 0,
        # "class_3": 0,
    }
    
    for i in obj:
        if i == 0: 
            count_dict['class_1'] += 1
        elif i == 1: 
            count_dict['class_2'] += 1
        # elif i == 2: 
        #     count_dict['class_3'] += 1              
        else:
            print("Check classes.")
            
    return count_dict


# classes are not identical in distribution, so use weighted rndom sampler, first get labels from train iterable
target_list = []
for _, t in train_dataset:
    target_list.append(t)
    
target_list = torch.tensor(target_list)
target_list = target_list[torch.randperm(len(target_list))]

# get counts of each class and divide to get weights of each class
class_count = [i for i in get_class_distribution(y_train).values()]
class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
print(class_weights)

# create list with weights in for target list vals now
class_weights_all = class_weights[target_list]

# instantiate Weighted random sampler to deal with disproportionate dist of classes (not that disproportionate), use weights list, number data points (70)
weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)


EPOCHS = 35 #100 overfits but 35 is about trough of validation loss
BATCH_SIZE = 4
LEARNING_RATE = 0.001
NUM_FEATURES = X_array.shape[1]
NUM_CLASSES = 2


train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)


if (torch.cuda.is_available()):
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


model = Lib_ANN.ML_Models(X_array.shape[1]) # change in and out features size depending on data used and type of classsification
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)


# compute accuracy, send y_pred from model into a log_softmax, which floors or celings to nearest prediction, chooses one with best prob
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}


print("Begin training.")
# incorporate k-fold cross validation here (LOOCV is just k-fold with fold # = sample size), as loop that treats kth fold as val set, then go back and average results over every fold for a given sample size
for e in range(1, EPOCHS+1):
    
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        
        y_train_pred = model(X_train_batch)
        
        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)
        
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
        
        
    # VALIDATION    
    with torch.no_grad():
        
        val_epoch_loss = 0
        val_epoch_acc = 0
        
        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = model(X_val_batch)
                        
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)
            
            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
                              
    
    print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')


# Create dataframes
train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
# Plot the dataframes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
plt.show()


y_pred_list = []
y_proba_list = []
with torch.no_grad():
    model.eval()
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        _, y_pred_tags = torch.max(y_test_pred, dim = 1)
        probabilities = nn.functional.softmax(y_test_pred, dim=1)[:, 1]
        y_pred_list.append(y_pred_tags.cpu().numpy())
        y_proba_list.append(probabilities.detach().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
y_proba_list = [a.squeeze().tolist() for a in y_proba_list]


confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list))

sns.heatmap(confusion_matrix_df, annot=True)

print(classification_report(y_test, y_pred_list))

plt.show()

#roc and auc score

plt.figure()
fpr, tpr, thresholds = roc_curve(y_test, y_proba_list, pos_label=1)
auc = roc_auc_score(y_test, y_proba_list)
plt.plot(fpr, tpr, linewidth=4, label='AUC= %.2f'% auc)

print('The AUC for {} is {}'.format(1, auc))

plt.plot([0, 1], [0, 1], 'k--') 
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()