#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-01 13:15:09
# @Author  : Jun Fu (fujun@mail.ustc.edu.cn)
# @Version : $Id$

from torch.utils.data import Dataset, DataLoader
import numpy as np 
import torch

class sinDataset(Dataset):
	""" sin(x) """

	def __init__(self, pointNums):

		super(sinDataset, self).__init__()

		self.pointNums = pointNums

	def __len__(self):

		return self.pointNums

	def __getitem__(self, idx):

		point = idx*2*3.14/self.pointNums

		y = np.sin(point)

		sample = {'point':np.array([point]), 'label': np.array([y])}

		# to tensor #

		sample['point'] = torch.from_numpy(sample['point']).float()

		sample['label'] = torch.from_numpy(sample['label']).float()

		return sample

class simDataLoader(object):

	def __init__(self, trainNums, testNums):

		super(simDataLoader, self).__init__()

		self.trainSet = sinDataset(trainNums)

		self.testSet = sinDataset(testNums)

		self.trainLoader = DataLoader(self.trainSet, batch_size=32, shuffle=True, num_workers=2)

		self.testLoader = DataLoader(self.trainSet, batch_size=32, shuffle=True, num_workers=2)

	def getLoader(self):

		return self.trainLoader, self.testLoader

	def getDataSet(self):

		return self.trainSet, self.testSet

