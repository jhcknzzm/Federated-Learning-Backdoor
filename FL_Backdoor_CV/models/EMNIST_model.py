import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import GroupNorm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = torch.nn.GroupNorm(32,32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = torch.nn.GroupNorm(32,64)
        self.conv2_drop = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(9216, 128)  ### 9216
        self.fc2 = nn.Linear(128, 47)

    # def forward(self, x):
    #     x = F.relu(self.bn1(self.conv1(x)))
    #     x = F.relu(self.bn2(self.conv2(x)))
    #     # print(x.size())
    #     x = x.view(-1, 36864)
    #     x = F.relu(self.fc1(x))
    #     # x = F.dropout(x, p=0.5)
    #     x = self.fc2(x)
    #     return x

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2_drop(F.max_pool2d(self.conv2(x), 2)))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # print(x.size())
        x = x.view(-1, 9216)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        return x