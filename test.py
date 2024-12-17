###
# test drive models with dummy inputs

import torch
import torch.nn as nn

### local imports
from models import LeNet5, LeNet5v2, LeNet5v2b


model = LeNet5()

#####
### single input (channels,height,width) => (1,32,32)
input = torch.randn(1,32,32)
print(f"{input.shape} {input.ndim}d", input)

output = model(input)
print(f"{output.shape} {output.ndim}d", output)

###
###  try batch of size one => (1,1,32,32)
print("---")
input = input.unsqueeze(0)
print(f"{input.shape} {input.ndim}d", input)

output = model(input)
print(f"{output.shape} {output.ndim}d", output)


###
###  try batch of size of two  => (2,1,32,32)
print("---")
input = torch.stack( (torch.randn(1,32,32),torch.randn(1,32,32)) )
print(f"{input.shape} {input.ndim}d", input)

output = model(input)
print(f"{output.shape} {output.ndim}d", output)


print( "bye")
