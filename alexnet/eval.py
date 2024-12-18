import torch
from torchvision import transforms
from PIL import Image


### local imports
from model import model



path = './files/coffee.jpg'
# path = './files/cat.jpg'
# path = './files/stephansdom.jpg'


img = Image.open( path )


transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),   # 224x224
  transforms.ToTensor(),
  transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])])

img_tensor = transform(img)
batch = torch.unsqueeze(img_tensor, dim=0)


model.eval()
y = model(batch)

y_max, index = torch.max(y,dim=1)

with open('./files/imagenet_class_labels.txt') as f:
  classes = [line.strip() for line in f.readlines()]

prob = torch.nn.functional.softmax(y, dim=1)[0]
print( classes[index[0]], prob[index[0]].item())
print()
#=> 967: 'espresso', 0.8799548745155334

y_sort, indices = torch.sort(y, descending=True)
for idx in indices[0][:5]:
  print(classes[idx], prob[idx].item())
#=> 967: 'espresso',     0.8799548745155334
#   968: 'cup',          0.07688959687948227
#   504: 'coffee mug',   0.038615722209215164
#   925: 'consomme',     0.0035129631869494915
#   960: 'chocolate sauce, chocolate syrup', 0.0005007769796065986


print( "bye" )


"""
try with

    path = './files/cat.jpg'

resulting in

    283: 'Persian cat',       0.31462812423706055
    552: 'feather boa, boa',  0.21569392085075378
    285: 'Egyptian cat',      0.17547936737537384
    281: 'tabby, tabby cat',  0.03902266174554825
    262: 'Brabancon griffon', 0.031412456184625626

or with

    path = './files/stephansdom.jpg'

resulting in

    497: 'church, church building', 0.4236260652542114
    698: 'palace',                  0.2292090207338333
    663: 'monastery',               0.1675940901041031
    442: 'bell cote, bell cot',     0.03244597092270851
    483: 'castle',                  0.024614153429865837
"""

