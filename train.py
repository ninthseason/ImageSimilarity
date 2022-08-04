import torch
from torch.optim import Adam, ASGD, SGD
from torch.nn import MSELoss, CrossEntropyLoss, HingeEmbeddingLoss
from torchmetrics import HingeLoss
from model import Siamese
from dataset import dataloader_train, dataloader_valid
from valid import valid

LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
EPOCH = 100
DEVICE = "cuda:0"
model = Siamese().to(DEVICE)
# try:
#     model.load_state_dict(torch.load("models/model_state.pt"))
# except FileNotFoundError:
#     pass

optimizer = SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
loss_fn = HingeLoss()

while EPOCH > 0:
    EPOCH -= 1
    model.train()
    total_loss = 0
    total_times = 0
    for idx, (pic1, pic2, y) in enumerate(dataloader_train):

        pic1 = pic1.to(DEVICE)
        pic2 = pic2.to(DEVICE)
        y = y.type(torch.FloatTensor)
        y = y.to(DEVICE)

        y_hat = model(pic1, pic2)
        y_hat = y_hat.view(-1)
        # print(y.view(-1))
        # print(y_hat.view(-1))
        loss = loss_fn(y_hat, y)
        # print(loss)
        # mask = y == y_hat
        # print(torch.count_nonzero(mask).item())
        # exit()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            # print(y_hat)
            total_loss += loss.item()
            total_times += 1
    print("epoch {}. total_loss: {} accuracy: {}".format(EPOCH, total_loss / total_times, valid(model, dataloader_valid)))
    torch.save(model.state_dict(), "models/model_state.pt")
