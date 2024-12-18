####
#  alexnet model via torchvision including weights


import torch
from torchvision import models
from torchvision.models import AlexNet_Weights


model = models.alexnet( weights=AlexNet_Weights.IMAGENET1K_V1 )





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


    print_model( model,  input_size=(3, 224, 224) )
    print("bye")

