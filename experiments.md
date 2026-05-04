# stuff we tried


## resnet 
8 epochs: 65 percent accuracy

## cnn

pimped datacamp cnn: 8 epochs only 63 percent

30 epochs with basic data augmentations: 
Epoch 29 - Val Loss: 0.9382192168054702, Val Acc: 0.6701, Val PCAcc: 0.6699157058604271

### adjusting channel sizes from 8 to 32 in the first layer and then subsequently to 128 in layer 3
Epoch 29 - Val Loss: 0.5740807143193257, Val Acc: 0.8027, Val PCAcc: 0.8028436010504292


### adding image rotation in augmentation: 




## vit

base performance at 8 epochs: 60 percent

30 epochs with data augmentation:  
Epoch 29 - Val Loss: 0.9428194625468194, Val Acc: 0.6747, Val PCAcc: 0.6746506419726448