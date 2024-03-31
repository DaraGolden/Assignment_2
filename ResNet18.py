import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms


class ResNet18(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        self.ResNet18_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224))
            ])

    def forward(self, x):
        return self.model(x)

    def transform(self, x):
        return self.ResNet18_transform(x)