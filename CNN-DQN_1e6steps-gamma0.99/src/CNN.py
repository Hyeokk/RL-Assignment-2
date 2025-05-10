import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNActionValue(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.relu):
        super(CNNActionValue, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)  # [N, 4, 84, 84] -> [N, 16, 20, 20]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # [N, 16, 20, 20] -> [N, 32, 9, 9]
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)  # [N, 32, 9, 9] -> [N, 64, 7, 7]
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # [N, 64, 7, 7] -> [N, 64, 5, 5]
        
        # 마지막 컨볼루션 레이어 이후의 특징 크기 계산
        self.in_features = 64 * 5 * 5
        
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = x.view((-1, self.in_features))
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x