import torch
import torch.nn as nn
import torch.nn.functional as F

class CurvedRoadCNN(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.relu):
        super(CurvedRoadCNN, self).__init__()
        
        # 1. 기본 컨볼루션 레이어 - 원래 구조 유지
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)  # [N, 4, 84, 84] -> [N, 16, 20, 20]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # [N, 16, 20, 20] -> [N, 32, 9, 9]
        
        # 2. 다중 스케일 특징 추출 - 복잡한 곡선 감지에 도움
        # 2.1 병렬 컨볼루션 브랜치 1 (기존 경로)
        self.conv3a = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  # [N, 32, 9, 9] -> [N, 32, 9, 9]
        self.conv4a = nn.Conv2d(32, 32, kernel_size=3, stride=1)  # [N, 32, 9, 9] -> [N, 32, 7, 7]
        
        # 2.2 병렬 컨볼루션 브랜치 2 (더 넓은 수용 영역)
        self.conv3b = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)  # [N, 32, 9, 9] -> [N, 32, 9, 9]
        self.conv4b = nn.Conv2d(32, 32, kernel_size=3, stride=1)  # [N, 32, 9, 9] -> [N, 32, 7, 7]
        
        # 3. 딜레이티드 컨볼루션 - 더 큰 수용 영역 (더 멀리 보기)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=2, padding=2)  # [N, 64, 7, 7] -> [N, 64, 7, 7]
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # [N, 64, 7, 7] -> [N, 64, 5, 5]
        
        # 4. 공간적 주의 메커니즘 - 중요한 도로 영역에 집중
        self.attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 마지막 컨볼루션 레이어 이후의 특징 크기 계산
        self.in_features = 64 * 5 * 5
        
        # 5. 강화된 완전 연결 레이어
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
        # 6. 드롭아웃 - 일반화 능력 향상
        self.dropout = nn.Dropout(0.2)
        
        self.activation = activation
        
        # 7. 배치 정규화 - 학습 안정화
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3a = nn.BatchNorm2d(32)
        self.bn3b = nn.BatchNorm2d(32)
        self.bn4a = nn.BatchNorm2d(32)
        self.bn4b = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)

    def forward(self, x):
        # 초기 컨볼루션 레이어
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        
        # 병렬 브랜치 - 다른 스케일에서 특징 추출
        x_a = self.activation(self.bn3a(self.conv3a(x)))
        x_a = self.activation(self.bn4a(self.conv4a(x_a)))
        
        x_b = self.activation(self.bn3b(self.conv3b(x)))
        x_b = self.activation(self.bn4b(self.conv4b(x_b)))
        
        # 두 브랜치 특징 연결
        x = torch.cat([x_a, x_b], dim=1)  # [N, 64, 7, 7]
        
        # 딜레이티드 컨볼루션으로 더 넓은 컨텍스트 캡처
        x = self.activation(self.bn5(self.conv5(x)))
        x = self.activation(self.bn6(self.conv6(x)))
        
        # 주의 메커니즘 적용
        attention_map = self.attention(x)
        x = x * attention_map  # 중요한 영역에 가중치 부여
        
        # 분류 레이어
        x = x.view((-1, self.in_features))
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        
        return x