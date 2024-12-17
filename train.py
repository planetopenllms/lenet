import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

### local imports
from models import LeNet5, LeNet5v2, LeNet5v2b


###
# setup train & test data (loaders)

# Define the transform for MNIST data (resize and normalize)
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to 32x32 (LeNet-5 requirement)
    transforms.ToTensor(),        # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform)


train_loader  = DataLoader(train_dataset, batch_size=64,   shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=1000, shuffle=False)

print( len(train_loader), len(test_loader))
# train - 60000 images in 938 batches (of 64)
# test    10000 images in 10  batches (of 1000)



# Create the model  - pick one of LeNet5, LetNet5v2, LeNet5v2b
model = LeNet5()


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
eval_iter = 100
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % eval_iter == 0:
            print( f"{epoch+1}/{num_epochs} - batch {batch_idx}/{len(train_loader)} - loss {loss.item()} - running_loss {running_loss/(batch_idx+1)}, total {total}")


    # Print statistics after each epoch
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    print(f"\n==> Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    # Evaluate on the test set after each epoch
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No need to calculate gradients during evaluation
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%\n")

# ==> Epoch [10/10], Loss: 0.0159, Accuracy: 99.52%
# Test Accuracy: 98.59%

# Save the trained model
torch.save(model.state_dict(), './lenet5.pth')
##  about 250 154 bytes  - lenet5.pth


print("bye")

