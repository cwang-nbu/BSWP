# BALANCED STRIPE-WISE PRUNING IN THE FILTER (BSWP)
![image](https://github.com/ajdt1111/BSWP/blob/main/framework.png)
## Introduction
    This work has been submitted to ICASSP 2022
## Requirements
python3.6 <br>
pytorch1.8.0 <br>
torchvision0.9.0 <br>
## Getting Started
#### Training and pruning ResNet56 with piecewise function
    python main_resnet.py --arch ResNet56 --save checkpoint/ResNet56 --use_function piecewise
#### Training and pruning ResNet56 with piecewise function
    python main_resnet.py --arch ResNet56 --save checkpoint/ResNet56 --use_function convex
#### Training and pruning VGG16
    python main_vgg.py --arch VGG --save checkpoint/VGG --use_function piecewise
## References
Our work is based on : Pruning Filter in Filter (https://arxiv.org/abs/2009.14410) <br>
Our code is alse based on https://github.com/fxmeng/Pruning-Filter-in-Filter <br>
