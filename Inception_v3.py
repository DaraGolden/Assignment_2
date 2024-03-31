import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


class Inceptionv3(nn.Module):
    def __init__(self):
        super(Inceptionv3, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        # Freeze parameters
        self.model.aux_logits = False
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Modify the fully connected layers
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
        
        self.shallow_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((299, 299))
            ])

    def forward(self, x):
        return self.model(x)

    def transform(self, x):
        return self.shallow_transform(x)