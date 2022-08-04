import torch
from torch import nn


class Siamese(nn.Module):
    def __init__(self):
        super().__init__()
        self.descriptor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decision = nn.Sequential(
            nn.Linear(32 * 16 * 16, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1),
            nn.Tanh()
        )

    def forward(self, patch1, patch2):
        # print(torch.count_nonzero(patch1 == patch2))
        patch1 = self.descriptor(patch1)
        patch2 = self.descriptor(patch2)
        # print("0的数量", torch.count_nonzero(patch1))
        # print("数据总和", torch.sum(patch1))
        # print(torch.count_nonzero(patch1 == patch2))
        patch1 = patch1.view(patch1.size(0), 16 * 16 * 16)
        patch2 = patch2.view(patch2.size(0), 16 * 16 * 16)
        x = torch.cat([patch1, patch2], dim=1)
        # print(torch.count_nonzero(patch1[0]))
        # print(patch1[0].size())
        # print(torch.count_nonzero(patch2[0]))
        # print(patch2[0].size())
        y = self.decision(x)
        return y
