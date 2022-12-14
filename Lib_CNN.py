import torch
import torch.nn as nn

class DataAccess():

    # This library is used to refer the data index and load the data during training

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index, :], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

class ML_Models(torch.nn.Module):

    def __init__(self, in_dim):

        super(ML_Models, self).__init__()
        # super here is used to inherit the properties from torch.nn.Module

        # Define the layers
        D_in = in_dim
        H1 = 128
        H2 = 64
        D_out = 2

        self.layer1 = nn.Linear(D_in, H1)
        self.layer2 = nn.Linear(H1, H2)
        self.layer3 = nn.Linear(H2, D_out)
        self.relu = nn.ReLU()

    def forward(self, x):

        y_t = self.relu(self.layer1(x))
        y_t = self.relu(self.layer2(y_t))
        output = self.layer3(y_t)

        return output


class ML_Models_Binary(torch.nn.Module):

    def __init__(self):

        super(ML_Models_Binary, self).__init__()
        # super here is used to inherit the properties from torch.nn.Module

        # Define the layers
        D_in = 3600
        H1 = 128
        H2 = 64
        D_out = 1

        self.layer1 = nn.Linear(D_in, H1)
        self.layer2 = nn.Linear(H1, H2)
        self.layer3 = nn.Linear(H2, D_out)
        self.relu = nn.ReLU()

    def forward(self, x):

        y_t = self.relu(self.layer1(x))
        y_t = self.relu(self.layer2(y_t))
        output = self.layer3(y_t)

        return output

class Raman_CNN(torch.nn.Module):
    def __init__(self, numChannels, classes, inputdim): #here numchannels is input dimension or wavelengths in spectra
        super(Raman_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=numChannels, out_channels=16, kernel_size= 12)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size= 12)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size= 12)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size= 12)

        self.fc = nn.LazyLinear(500) #all wavelengths * numchannels? Lazy Linear infers in_features
        self.fc2 = nn.Linear(500, classes) #1 if binary cross entropy
        self.dropout = nn.Dropout(p=.1)
        self.input_dim = inputdim
    
    def forward(self, x):
        # print(x.shape)
        x = x.view(x.shape[0], 1, self.input_dim) #do this to get it into shape of batch, channels, dimension of spectra
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.maxpool(self.relu(self.conv3(x)))
        x = self.maxpool(self.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.dropout(self.fc(x)))
        out = self.fc2(x)

        return out