# BALANCED STRIPE-WISE PRUNING IN THE FILTER (BSWP)
![image](https://github.com/ajdt1111/BSWP/blob/main/framework.png)
#### Requirements
python3.6 <br>
pytorch1.8.0 <br>
torchvision0.9.0 <br>
## Getting Started
### Training and pruning ResNet56
python main_resnet.py --arch ResNet56 --save checkpoint/ResNet56 <br>
### Training and pruning VGG16
python main_vgg.py --arch VGG --save checkpoint/VGG <br>
## References
our work is based on: Pruning Filter in Filter (https://arxiv.org/abs/2009.14410) <br>
our code is alse based on https://github.com/fxmeng/Pruning-Filter-in-Filter <br>
