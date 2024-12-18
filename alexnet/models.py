####
#  alexnet models


import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights

## pre-built via torchvision  including weights
model = models.alexnet( weights=AlexNet_Weights.IMAGENET1K_V1 )



####
#  do-it-yourself version via
#     https://d2l.ai/chapter_convolutional-modern/alexnet.html


class AlexNet(nn.Module):
    def __init__(self,num_classes=1000):
        super(AlexNet, self).__init__()

        self.layers = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        # Forward pass through the network
        return self.layers( x )




if __name__ == '__main__':
    # Print the model summaries
    from torchsummary import summary

    def print_model( model, input_size ):
      print(  "="*20,
             f"\n=  {model.__class__.__name__}  input_size={input_size}" )

      print()
      print(model)

      num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
      print("Total number of trainable model parameters:", num_params)
      total_bytes = num_params * 4   # assume float32 (4 bytes)
      print( f"about  {total_bytes / (1028 *1028) :.2f} MBs, {total_bytes / 1028 :.2f} KBs" )

      print( "\nsummary:")
      summary(model, input_size)


    print_model( model,      input_size=(3, 224, 224) )

    model = AlexNet()
    x = torch.randn(3,224,224).unsqueeze(0)
    y = model.forward( x )
    ## note - (auto-)summary not working for now (not working with Lazy) e.g.
    ##  ValueError: Attempted to use an uninitialized parameter in
    ##   <method 'numel' of 'torch._C.TensorBase' objects>.
    ##   This error happens when you are using a `LazyModule`
    ##   or explicitly manipulating `torch.nn.parameter.UninitializedParameter` objects.
    ##   When using LazyModules Call `forward` with a dummy batch to initialize
    ##   the parameters before calling torch functions
    print_model( model,  input_size=(3, 224, 224) )
    print("bye")

