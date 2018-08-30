# 使用h5py存储cifar10的数据集图片
## h5py简介
### HDF5
HDF5是用于存储和管理数据的数据模型，库和文件格式。它支持无限种类的数据类型，旨在实现灵活高效的I/O以及高容量和复杂数据。HDF5是可移植的，并且是可扩展的，允许应用程序在使用HDF5时不断发展。
### h5py
h5py是操作和使用HDF5数据的常用包，它是HDF5二进制数据格式的Python接口。  
一个h5py文件是 “dataset” 和 “group” 两种对象的容器。
- dataset : 类似数组的数据集合，像numpy数组一样工作 
- group : 像文件夹一样的容器，存放dataset和其他group，像字典一样工作

**cifar10的h5文件下载地址：**  
[csdn](https://download.csdn.net/download/qq_35206320/10630923)