import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Feature extraction
        # 3 layers of convolutional layers with ReLU activation and max pooling
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5), # -> [32, 296, 296]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # -> [32, 148, 148]

            nn.Conv2d(32, 64, kernel_size=5), # -> [64, 144, 144]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # -> [64, 72, 72]

            nn.Conv2d(64, 128, kernel_size=3),  # -> [128, 70, 70]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) # -> [128, 35, 35]
        )

        # fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(128 * 35 * 35, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )

    def forward_once(self, x):
        # sigle forward pass for one input
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        # calculate feature vectors for both inputs
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
