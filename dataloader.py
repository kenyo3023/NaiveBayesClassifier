#encoding=utf8
import numpy as np
import pandas as pd
import struct
import os

from PIL import Image
import matplotlib.pyplot as plt

class Dataloader():
    def __init__(self, data_path, label_path):
        self.data_path = data_path
        self.label_path = label_path
        self.data_offset, self.labels_offset = 16, 8
        
    def read_data(self):
        fd = open(self.data_path , 'rb')
        buf = fd.read()
        fd.close() 
        magic, numImages , numRows , numColumns = struct.unpack_from('>IIII', buf, 0)
        offset = self.data_offset
        data = []
        for i in range(0, numImages): 
            im = struct.unpack_from('>784B' ,buf, offset)
            offset += struct.calcsize('>784B')
            im = np.array(im,dtype='uint8')
            im = im.reshape(28,28)
            data.append(im)
        self.data = np.array(data)
    
    def read_labels(self):
        fd = open(self.label_path , 'rb')
        buf = fd.read()
        fd.close() 
        magic, numImages , numRows , numColumns = struct.unpack_from('>IIII', buf, 0)
        offset = self.labels_offset        
        labels = []
        for i in range(0, numImages): 
            im = struct.unpack_from('B' ,buf, offset)
            offset += struct.calcsize('B')
            im = np.array(im,dtype='uint8')
            labels.append(im)
        self.labels = np.array(labels)
    
    def display(self, index):
        im = self.data[index]
        plt.imshow(im, cmap = plt.cm.gray)                                        
        plt.show()
        print('index: ', self.labels[index][0])
        
    def save(self, save_path, X_name, y_name):
        np.save(os.path.join(save_path, X_name), self.data)
        np.save(os.path.join(save_path, y_name), self.labels)


if __name__ == '__main__':
    file_list = {                                                                   
                'X_train': 'train-images.idx3-ubyte',                                          
                'y_train': 'train-labels.idx1-ubyte',                                          
                'X_test': 't10k-images.idx3-ubyte',                                           
                'y_test': 't10k-labels.idx1-ubyte',                                           
                }
    X_train = os.path.join('dataset', file_list['X_train'])
    y_train = os.path.join('dataset', file_list['y_train'])
    train = Dataloader(X_train, y_train)
    train.read_data()
    train.read_labels()

    X_test = os.path.join('dataset', file_list['X_test'])
    y_test = os.path.join('dataset', file_list['y_test'])
    test = Dataloader(X_test, y_test)
    test.read_data()
    test.read_labels()

    path = 'dataset/mnist'
    if not os.path.isdir(path):
        os.mkdir(path)
    train.save(path, 'X_train.npy', 'y_train.npy')
    print('train save done')
    test.save(path, 'X_test.npy', 'y_test.npy')
    print('test save done')

    X_train = np.load('dataset/mnist/X_train.npy').reshape(-1, 28*28)
    y_train = np.load('dataset/mnist/y_train.npy').reshape(60000, 1)
    X_test = np.load('dataset/mnist/X_test.npy').reshape(-1, 28*28)
    y_test = np.load('dataset/mnist/y_test.npy').reshape(10000, 1)

    # continous to categories
    interval = round(255/32)
    interval_group = [i*interval for i in range(round(255/interval)+1)]
    interval_index = [i for i in range(len(interval_group)-1)]

    def con2cat(X, group, index):
        m = X.shape[1]
        X = X.reshape(-1)
        X = np.array(pd.cut(X, group, right=False, labels=index))
        X = X.reshape(-1, m)
        return X

    X_train_cat = con2cat(X_train, interval_group, interval_index)
    X_test_cat = con2cat(X_test, interval_group, interval_index)

    np.save('dataset/mnist/X_train_cat.npy', X_train_cat)
    print('train cat save done')
    np.save('dataset/mnist/X_test_cat.npy', X_test_cat)
    print('test cat save done')