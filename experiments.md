# stuff we tried


## resnet 
8 epochs: 65 percent accuracy


Base case: 
model_bestResNet18_adamw_lr1e-03_exponential=0.9_30ep_bs128_ToImage-ToDtype-Normalize.pt

massive overfitting

base case:

Test loss: 1.6751
Overall Accuracy: 0.7618
Mean Per-Class Accuracy: 0.7618
Per-Class Accuracies:
  plane: 0.8220
  car: 0.8440
  bird: 0.6760
  cat: 0.6040
  deer: 0.7360
  dog: 0.6190
  frog: 0.8210
  horse: 0.7880
  ship: 0.8800
  truck: 0.8280

best data augmentation:

Test loss: 0.5657
Overall Accuracy: 0.8208
Mean Per-Class Accuracy: 0.8208
Per-Class Accuracies:
  plane: 0.8680
  car: 0.8840
  bird: 0.7570
  cat: 0.6510
  deer: 0.8290
  dog: 0.7090
  frog: 0.8670
  horse: 0.8470
  ship: 0.9060
  truck: 0.8900

best weight decay

Test loss: 0.9668
Overall Accuracy: 0.7837
Mean Per-Class Accuracy: 0.7837
Per-Class Accuracies:
  plane: 0.8320
  car: 0.8750
  bird: 0.7060
  cat: 0.5560
  deer: 0.7940
  dog: 0.6930
  frog: 0.8290
  horse: 0.8350
  ship: 0.8770
  truck: 0.8400

best weight decay + data augmentation

Test loss: 0.4803
Overall Accuracy: 0.8483
Mean Per-Class Accuracy: 0.8483
Per-Class Accuracies:
  plane: 0.8970
  car: 0.9100
  bird: 0.7970
  cat: 0.7140
  deer: 0.8350
  dog: 0.7150
  frog: 0.9100
  horse: 0.9040
  ship: 0.9190
  truck: 0.8820


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


best resnet

Test loss: 0.4941
Overall Accuracy: 0.8480
Mean Per-Class Accuracy: 0.8480
Per-Class Accuracies:
  plane: 0.8540
  car: 0.9190
  bird: 0.8170
  cat: 0.6950
  deer: 0.8370
  dog: 0.7600
  frog: 0.8930
  horse: 0.8830
  ship: 0.9280
  truck: 0.8940

best Cnn

Test loss: 0.5710
Overall Accuracy: 0.8019
Mean Per-Class Accuracy: 0.8019
Per-Class Accuracies:
  plane: 0.8470
  car: 0.9020
  bird: 0.7090
  cat: 0.6120
  deer: 0.7900
  dog: 0.7230
  frog: 0.8400
  horse: 0.8350
  ship: 0.9010
  truck: 0.8600

best ViT

Test loss: 0.7694
Overall Accuracy: 0.7353
Mean Per-Class Accuracy: 0.7353
Per-Class Accuracies:
  plane: 0.8040
  car: 0.8480
  bird: 0.6290
  cat: 0.6290
  deer: 0.6330
  dog: 0.6180
  frog: 0.8120
  horse: 0.8180
  ship: 0.8040
  truck: 0.7580
