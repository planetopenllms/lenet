####
# different variants of LeNet v5 & friends

import torch
import torch.nn as nn
import torch.nn.functional as F


###
#  note: expects an input_size of 32x32 for the image!!
#          if you use 28x28 for the image you MUST add padding to conv1
#                 to make sure the output shape is 28x28!!

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

       # Layer 1: Convolutional layer with 6 filters of size 5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # Subsampling (average pooling)
        # Layer 2: Convolutional layer with 16 filters of size 5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # Subsampling (average pooling)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Flattened output from previous layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # Output layer with 10 classes

    def forward(self, x):
        # Forward pass through the network
        x = self.pool1(F.tanh(self.conv1(x)))  # Conv1 -> Tanh -> Pooling
        x = self.pool2(F.tanh(self.conv2(x)))  # Conv2 -> Tanh -> Pooling
        x = x.view(-1, 16 * 5 * 5)  # Flatten the output for fully connected layers
        x = F.tanh(self.fc1(x))  # FC1 -> Tanh
        x = F.tanh(self.fc2(x))  # FC2 -> Tanh
        x = self.fc3(x)  # FC3 (Output layer)
        return x


###############
#  alt. version with nn.Sequentail
#  note - input x MUST always be batch e.g (1,1,32,32) NOT (1,32,32)
#      nn.Flatten will NOT work on single inputs
#      resulting in
#        RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x25 and 400x120)

class LeNet5Seq(nn.Module):
    def __init__(self):
        super(LeNet5Seq, self).__init__()

        self.layers = nn.Sequential(
            # Layer 1: Convolutional layer with 6 filters of size 5x5
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # Subsampling (avg pooling)
            # Layer 2: Convolutional layer with 16 filters of size 5x5
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # Subsampling (avg pooling)
            nn.Flatten(),
            # Fully connected layers
            nn.Linear(16 * 5 * 5, 120),  # Flattened output from previous layer
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)    # Output layer with 10 classes
          )

    def forward(self, x):
        # Forward pass through the network
        return self.layers( x )



###
#  what's different?
#    1) change activation from tanh to relu
#    2) change avg pooling to max pooling

class LeNet5v2(nn.Module):
    def __init__(self):
        super(LeNet5v2, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # 1 input channel (grayscale), 6 output channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Subsampling (average pooling)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 6 input channels, 16 output channels
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Subsampling (average pooling)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)  # Fully connected layer
        self.fc3 = nn.Linear(84, 10)   # 10 output classes (for example, MNIST digits)

    def forward(self, x):
        # Apply layers with activation functions
        x = self.pool1(F.relu(self.conv1(x)))  # Convolution + ReLU
        x = self.pool2(F.relu(self.conv2(x)))  # Convolution + ReLU
        x = x.view(-1, 16 * 5 * 5)  # Flatten the output for fully connected layers
        x = F.relu(self.fc1(x))    # Fully connected + ReLU
        x = F.relu(self.fc2(x))
        x = self.fc3(x)            # Output layer (logits)
        return x


###
#  what's different?
#   -  add a third conv2d layer WITHOUT pooling
#         gets 16 channels with 5x5 input and
#              outputs 120 filters with kernel 5x5 => (120,1,1)
#   why?  let's ask / prompt the a.i. for an answer / reason

class LeNet5v2b(nn.Module):
    def __init__(self):
        super(LeNet5v2b, self).__init__()

        # Define the layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # 1 input channel (grayscale), 6 output channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Subsampling (average pooling)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 6 input channels, 16 output channels
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Subsampling (average pooling)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)  # 16 input channels, 120 output channels
        self.fc1 = nn.Linear(120, 84)  # Fully connected layer
        self.fc2 = nn.Linear(84, 10)   # 10 output classes (for example, MNIST digits)

    def forward(self, x):
        # Apply layers with activation functions
        x = self.pool1(F.relu(self.conv1(x)))  # Convolution + ReLU
        x = self.pool2(F.relu(self.conv2(x)))  # Convolution + ReLU
        x = F.relu(self.conv3(x))  # Convolution + ReLU
        x = x.view(-1, 120)         # Flatten the tensor
        x = F.relu(self.fc1(x))    # Fully connected + ReLU
        x = self.fc2(x)            # Output layer (logits)
        return x





###############
#  alt. version with padding 2 in conv1  for 28x28 input_size
#
#  note - input x MUST be batch e.g (1,1,28,28) NOT (1,28,28)
#      nn.Flatten will NOT work on single inputs
#      resulting in
#        RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x25 and 400x120)

class LeNet5_28x28(nn.Module):
    def __init__(self):
        super(LeNet5_28x28, self).__init__()

        self.layers = nn.Sequential(
            # Layer 1: Convolutional layer with 6 filters of size 5x5
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Subsampling (max pooling)
            # Layer 2: Convolutional layer with 16 filters of size 5x5
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Subsampling (max pooling)
            nn.Flatten(),
            # Fully connected layers
            nn.Linear(16 * 5 * 5, 120),  # Flattened output from previous layer
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)    # Output layer with 10 classes
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

      # Print model summary for a batch of 1 grayscale image (1x32x32 size)
      ## or use channel  e.g. (32, 32, 1) - why? why not?
      print( "\nsummary:")
      summary(model, input_size)


    print_model( LeNet5(),    input_size=(1,32,32) )
    print_model( LeNet5Seq(), input_size=(1,32,32) )
    print_model( LeNet5v2(),  input_size=(1,32,32) )
    print_model( LeNet5v2b(), input_size=(1,32,32) )

    print_model( LeNet5_28x28(), input_size=(1,28,28 ))
    print("bye")


