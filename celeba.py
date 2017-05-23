import numpy as np
import h5py
import os

class celeba:
	def __init__(self):
		file_name = '/tempspace/lcai/GAN/Data/data/celeba.h5'
		self.data = h5py.File(file_name,'r')['data']
		self.data_set= self.data[:160000,:,:,:]
		self.test_set= self.data[160000:,:,:,:]
		self.train_idx = 0
		self.test_idx= 0

	def next_batch(self, batch_size):
		prev_idx = self.train_idx
		self.train_idx += batch_size
		if self.train_idx > self.data_set.shape[0]:
			self.train_idx = batch_size
			prev_idx = 0

		return self.data_set[prev_idx:self.train_idx,:,:,:]

	def next_test_batch(self, batch_size):
		prev_idx = self.test_idx
		self.test_idx += batch_size
		return self.test_set[prev_idx:self.test_idx, : , :, :]
	
	def reset(self):
		self.test_idx =0