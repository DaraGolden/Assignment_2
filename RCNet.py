import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


# RCNET architecture implemented from the paper:
# Dewangan, Deepak Kumar, and Satya Prakash Sahu. 
# "RCNet: road classification convolutional neural networks for intelligent vehicle system." 
# Intelligent Service Robotics 14, no. 2 (2021): 199-214.

class RCNet(nn.Module):
    def __init__(self):
        super(RCNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout()
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout()
        self.fc1 = nn.Linear(12288, 256, bias=True)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256, bias=True)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 3, bias=True)
        
        self.RCNet_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((96, 64))
            ])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.dropout1(x)
        x = self.upsample1(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.bn1(x)
        x = self.pool4(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1) # Flatten layer
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        x = self.fc3(x)
        return x

    def transform(self, x):
        return self.RCNet_transform(x)




