import PIL.Image
import torch
from torchvision.transforms import ToTensor, Resize, Compose
from model import Siamese

model = Siamese()
model.eval()
model.load_state_dict(torch.load("models/model_state_30.pt"))
pic1 = PIL.Image.open("test/bottle1.jpg").convert("L")
pic2 = PIL.Image.open("test/desk.jpg").convert("L")
transform = Compose([Resize([64, 64]), ToTensor()])
pic1 = transform(pic1)
pic2 = transform(pic2)
pic1 = pic1.view(1, pic1.size(0), pic1.size(1), pic1.size(2))
pic2 = pic2.view(1, pic2.size(0), pic2.size(1), pic2.size(2))

output = model(pic1, pic2)
print(output)