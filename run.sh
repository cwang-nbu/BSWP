train(){
  python main_vgg.py \
  --data_path ./cifar-10 \
  --save ./checkpoint/VGG
}

train();


