import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

### local imports
from models import LeNet5, LeNet5v2, LeNet5v2b


# Create the model and  load the (trained) weights
model = LeNet5()
model.load_state_dict(torch.load('./lenet5.pth'))


# Define the transform for MNIST data (resize and normalize)
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to 32x32 (LeNet-5 requirement)
    transforms.ToTensor(),        # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images
])

# Load the MNIST dataset
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)



# Evaluate on the test set after each epoch
model.eval()  # Set the model to evaluation mode
correct = 0
total   = 0
with torch.no_grad():  # No need to calculate gradients during evaluation
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%\n")


#=> Test Accuracy: 98.59%


print("bye")