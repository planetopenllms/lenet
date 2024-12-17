###
# download the MNIST dataset
#   if not already downloaded
#
# /data
#  └───MNIST
#     └───raw
#             t10k-images-idx3-ubyte
#             t10k-images-idx3-ubyte.gz
#             t10k-labels-idx1-ubyte
#             t10k-labels-idx1-ubyte.gz
#             train-images-idx3-ubyte
#             train-images-idx3-ubyte.gz
#             train-labels-idx1-ubyte
#             train-labels-idx1-ubyte.gz


from torchvision import datasets

train_dataset = datasets.MNIST(root='./data', train=True, download=True )
print( train_dataset )
# Dataset MNIST
#    Number of datapoints: 60000
#    Root location: ./data
#    Split: Train
test_dataset = datasets.MNIST(root='./data', train=False, download=True)
print( test_dataset )
# Dataset MNIST
#    Number of datapoints: 10000
#    Root location: ./data
#    Split: Test


print( train_dataset[0] )
# (<PIL.Image.Image image mode=L size=28x28 at 0x1D61C6F6210>, 5)
print( test_dataset[0] )
# (<PIL.Image.Image image mode=L size=28x28 at 0x1D61BE372C0>, 7)


print( "bye")

