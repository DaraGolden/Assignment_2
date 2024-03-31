import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms




# Implementation of: 
# Woo, Sanghyun, Jongchan Park, Joon-Young Lee, and In So Kweon. 
# "Cbam: Convolutional block attention module." 
# In Proceedings of the European conference on computer vision (ECCV), pp. 3-19. 2018.
# implemented by Peachypie98 on github at the following url https://github.com/Peachypie98/CBAM
class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output

class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x 
        return output 
    
class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x

# RCNET architecture implemented from the paper:
# Dewangan, Deepak Kumar, and Satya Prakash Sahu. 
# "RCNet: road classification convolutional neural networks for intelligent vehicle system." 
# Intelligent Service Robotics 14, no. 2 (2021): 199-214.

class RCNet_attention(nn.Module):
    def __init__(self):
        super(RCNet_attention, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.CBAM1 = CBAM(32, 4)  # CBAM attention module after conv2
        self.dropout1 = nn.Dropout(0.25)  # Dropout after pooling
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.CBAM2 = CBAM(64, 4)  # CBAM attention module after conv4
        self.dropout2 = nn.Dropout(0.25)  # Dropout after pooling
        self.conv5 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.CBAM3 = CBAM(128, 4)  # CBAM attention module after conv6
        self.dropout3 = nn.Dropout(0.25)  # Dropout after pooling
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.CBAM4 = CBAM(128, 4)  # CBAM attention module after conv8
        self.bn1 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(0.25)  # Dropout after pooling
        self.fc1 = nn.Linear(12288, 256, bias=True)
        self.dropout_fc1 = nn.Dropout(0.5)  
        self.bn2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256, bias=True)
        self.dropout_fc2 = nn.Dropout(0.5)  
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
        x = self.CBAM1(x)
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.CBAM2(x)
        x = self.dropout2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.CBAM3(x)
        x = self.dropout3(x)
        x = self.upsample1(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.CBAM4(x)
        x = self.bn1(x)
        x = self.pool4(x)
        x = self.dropout4(x)
        x = x.view(x.size(0), -1)  # Flatten layer
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        x = self.bn2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc2(x)
        x = self.bn3(x)
        x = self.fc3(x)
        return x

    def transform(self, x):
        return self.RCNet_transform(x)





