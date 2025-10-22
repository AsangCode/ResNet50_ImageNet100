import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_classes=100):  # Keeping default num_classes=100 from original
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=None)  # Initialize without pre-trained weights
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)