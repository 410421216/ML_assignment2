# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 06:02:27 2018

@author: Administrator
"""

from __future__ import print_function

import keras
from keras.datasets import mnist

#查看mnist資料型態
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#reducing dimension & processing character image
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train.shape)
print(x_test.shape)