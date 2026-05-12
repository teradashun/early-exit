import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28,400)
        self.fc2 = nn.Linear(400,200)
        self.fc3 = nn.Linear(200,100)
        self.fc4 = nn.Linear(100,10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,16,3,1)  # (input_channels, output_channels, Kernel_size, stride)
        self.pool = nn.MaxPool2d(2,2)  # (Kernel_size, stride)
        self.conv2 = nn.Conv2d(16,32,3,1)
        self.fc1 = nn.Linear(5*5*32,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 5*5*32)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5,1)  # (input_channels, output_channels, Kernel_size, stride)
        self.pool = nn.MaxPool2d(2,2)  # (Kernel_size, stride)
        self.conv2 = nn.Conv2d(6,16,5,1)
        self.fc1 = nn.Linear(5*5*16,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 5*5*16)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5,1)  # (input_channels, output_channels, Kernel_size, stride) 
        self.pool = nn.AvgPool2d(2,2)  # (Kernel_size, stride)
        self.conv2 = nn.Conv2d(6,16,5,1)
        self.conv3 = nn.Conv2d(16,120,4,1)
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,10)
    
    def forward(self, x):
        # (28x28x1 -> 24x24x6 -> 12x12x6)
        x = self.pool(F.tanh(self.conv1(x)))
        # (12x12x6 -> 8x8x16 -> 4x4x16)
        x = self.pool(F.tanh(self.conv2(x)))
        # (4x4x16 -> 1x1x120)
        x = F.tanh(self.conv3(x))
        x = x.view(-1, 120)
        # (1x1x120 -> 1x1x84)
        x = F.tanh(self.fc1(x))
        # (1x1x84 -> 1x1x10)
        x = self.fc2(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x