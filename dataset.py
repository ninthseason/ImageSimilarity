from torchvision.datasets import PhotoTour
from torchvision.transforms import ToTensor, ToPILImage, Compose
from torch.utils.data import DataLoader

dataset_train = PhotoTour("datasets", "liberty", train=False, transform=Compose([ToPILImage(), ToTensor()]), download=True)
dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)

dataset_valid = PhotoTour("datasets", "liberty", train=False, transform=Compose([ToPILImage(), ToTensor()]), download=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=128, shuffle=True)

print(len(dataloader_train))

if __name__ == '__main__':
    for pic1, pic2, i in dataloader_valid:
        print(pic1.size())
        print(pic2.size())
        print(i.size())
        break
