import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyLidarNet(nn.Module):
    """
    Tiny LIDAR-based policy network that incorporates the previous action for autoregressive control.

    Inputs:
      - lidar: Tensor of shape (batch, 1, L) where L is the number of lidar beams (e.g., 1080)
      - prev_action: Tensor of shape (batch, A) where A is the action dimension (e.g., 2)

    Output:
      - action: Tensor of shape (batch, A) with values in [-1, 1]
    """
    def __init__(self, lidar_dim: int = 1080, act_dim: int = 2):
        super().__init__()
        # 1D convolutional feature extractor
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, stride=1)
        # Pool to fixed length
        self.pool = nn.AdaptiveAvgPool1d(output_size=16)
        # Fully connected layers: conv features + previous action
        conv_feat_size = 64 * 16  # channels * pooled length
        self.fc1 = nn.Linear(conv_feat_size + act_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc_mu = nn.Linear(10, act_dim)

    def forward(self, lidar: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        # lidar: (batch, 1, L)
        x = F.relu(self.conv1(lidar))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)               # (batch, 64, 16)
        x = torch.flatten(x, 1)        # (batch, conv_feat_size)
        # concatenate previous action
        x = torch.cat([x, prev_action], dim=1)  # (batch, conv_feat_size + act_dim)
        # MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = torch.tanh(self.fc_mu(x))
        return mu

