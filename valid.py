import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import dataloader_valid
from model import Siamese

DEVICE = "cuda:0"


def valid(model: nn.Module, dataloader: DataLoader):
    model.eval()
    correct_number = 0
    total_number = 0
    for (pic1, pic2, y) in dataloader:
        pic1 = pic1.to(DEVICE)
        pic2 = pic2.to(DEVICE)
        y = y.to(DEVICE)
        y_hat = model(pic1, pic2)
        y_hat = torch.where(y_hat >= 0, 1, 0)
        y_hat = y_hat.view(y_hat.size(0))
        # print(torch.count_nonzero(y))
        # print(torch.count_nonzero(y_hat))
        mask = y == y_hat
        total_number += len(mask)
        correct_number += torch.count_nonzero(mask).item()
        # print(correct_number)
        break
    model.train()
    return correct_number / total_number


if __name__ == '__main__':
    model = Siamese().to(DEVICE)
    model.load_state_dict(torch.load("models/model_state.pt"))
    print(valid(model, dataloader=dataloader_valid))
