import torch
import torch.nn as nn
import torch.nn.functional as F


class B_LeNet_5(nn.Module):
    def __init__(self):
        super(B_LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5,1,2)  # (input_channels, output_channels, Kernel_size, stride)
        self.pool = nn.AvgPool2d(2,2)  # (Kernel_size, stride)
        self.conv2 = nn.Conv2d(6,16,5,1)
        self.conv3 = nn.Conv2d(16,120,5,1)
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,10)
        self.b_conv = nn.Conv2d(6,16,3,1)
        self.b_fc = nn.Linear(16*6*6,10)
    
    def forward(self, x, threshold=None):
        x = self.pool(F.tanh(self.conv1(x)))

        # 分岐1
        x_branch = self.pool(F.tanh(self.b_conv(x)))
        x_branch = x_branch.view(-1, 16*6*6)
        out_branch = self.b_fc(x_branch)
        if threshold is not None:
            probs_branch = F.softmax(out_branch, dim=1)
            entropy = torch.sum(-probs_branch * torch.log(probs_branch + 1e-9), dim=1)
            if torch.all(entropy <= threshold):
                return out_branch, None

        # main
        # (14x14x6 -> 10x10x16 -> 5x5x16)
        x = self.pool(F.tanh(self.conv2(x)))
        # (5x5x16 -> 1x1x120)
        x = F.tanh(self.conv3(x))
        x = x.view(-1, 120)
        # (1x1x120 -> 1x1x84)
        x = F.tanh(self.fc1(x))
        # (1x1x84 -> 1x1x10)
        out_main = self.fc2(x)
        return out_branch, out_main


class B_AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 出力: 16x16
            )
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 出力 8x8
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 出力 4x4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, num_classes),
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 出力 8x8
            )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 出力 4x4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, num_classes)
        )
        
        self.layer3_5 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 出力: 4x4
        )

        #  Main Classifier
        self.branch3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor, threshold=None) -> torch.Tensor:
        x1 = self.layer1(x)
        out1 = self.branch1(x1)
        if threshold is not None:
            probs_branch1 = F.softmax(out1, dim=1)
            entropy1 = torch.sum(-probs_branch1 * torch.log(probs_branch1 + 1e-9), dim=1)
            if torch.all(entropy1 <= threshold[0]):
                return out1, None, None

        x2 = self.layer2(x1)
        out2 = self.branch2(x2)
        if threshold is not None:
            probs_branch2 = F.softmax(out2, dim=1)
            entropy2 = torch.sum(-probs_branch2 * torch.log(probs_branch2 + 1e-9), dim=1)
            if torch.all(entropy2 <= threshold[1]):
                return out1, out2, None

        x3 = self.layer3_5(x2)
        x3 = torch.flatten(x3, 1)
        out3 = self.branch3(x3)

        return out1, out2, out3