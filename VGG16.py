import torch.nn as nn
from torchvision import models
from torchvision import datasets, transforms


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model = models.vgg16(pretrained=True)
        # Freeze early layers
        for param in self.model.parameters():
            param.requires_grad = False
        n_inputs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
        
        self.vgg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)) 
        ])

    def forward(self, x):
        return self.model(x)
    
    def transform(self, x):
        return self.vgg_transform(x)
        

