
mkdir data
mkdir data/cifar100
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzvf cifar-100-python.tar.gz -C ./data/cifar100
python cifar100-unpack.py

mkdir outputs
python VGG_CIFAR100.py
