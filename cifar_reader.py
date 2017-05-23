import numpy as np
import _pickle as cPickle
import os

class cifar_reader:
    def __init__(self):
        self.data_dir = '/tempspace/hyuan/VAE/Cifar/cifar-10-batches-py/'
        self.train_idx= 0
        self.test_idx =0
        self.data_set = self.load()
        self.data_set = np.transpose(self.data_set,(0,2,3,1) )
        self.data_set = self.data_set/127.5 - 1
        self.test_set =self.load_test()
        self.test_set = np.transpose(self.test_set,(0,2,3,1) )
        self.test_set = self.test_set/127.5 - 1

    def unpickle(self,file):
        fo = open(file, 'rb')
        data = cPickle.load(fo, encoding='latin1')
        fo.close()
        x= data['data'].reshape((10000,3,32,32))
        return x
    def load(self):
        train_data = [self.unpickle(self.data_dir+'data_batch_'+str(i)) for i in range(1,6)]
        train_x = np.concatenate([d for d in train_data], axis=0)
        np.random.seed(0)
        np.random.shuffle(train_x)
        return train_x

    def load_test(self):
        train_data = [self.unpickle(self.data_dir+'data_batch_'+str(i)) for i in range(6,7)]
        train_x = np.concatenate([d for d in train_data], axis=0)
        np.random.seed(0)
        np.random.shuffle(train_x)
        return train_x

    def next_batch(self, batch_size):
        prev_idx = self.train_idx
        self.train_idx += batch_size
        if self.train_idx> self.data_set.shape[0]:
            np.random.seed(0)
            np.random.shuffle(self.data_set)
            self.train_idx= batch_size
            prev_idx=0
        
        return self.data_set[prev_idx:self.train_idx, : , :, :]


    def next_test_batch(self , batch_size):
        prev_idx = self.test_idx
        self.test_idx += batch_size
        return self.test_set[prev_idx:self.test_idx, : , :, :]
        
    def reset(self):
        self.test_idx =0