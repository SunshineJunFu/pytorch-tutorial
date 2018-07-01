#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-01 13:36:52
# @Author  : Jun Fu (fujun@mail.ustc.edu.cn)
# @Version : $Id$

"""Attention

1. torch.device connot be used

2. SummaryWriter cannot be used ==> communicate with Queue solution

"""


import torch.nn as nn
import torch.multiprocessing as mp 
import time


class testWorker(mp.Process):

	def __init__(self, global_net, local_net, update_flag, lock, max_epoch, dataloader, finish, queue):

		super(testWorker, self).__init__()

		self.global_net = global_net

		self.update_flag = update_flag

		self.lock = lock

		self.max_epoch = max_epoch

		self.local_net = local_net

		self.dataloader = dataloader

		self.finish = finish

		self.queue = queue

		# self.writer = SummaryWriter('./logs/test/')


	def run(self):

		epoch_counter = 0

		global_counter = 0

		while True:

			self.lock.acquire()

			update_flag = self.update_flag.value

			self.lock.release()

			if update_flag == 0:

				time.sleep(0.001)

				continue

			self.lock.acquire()

			self.update_flag.value = 0 # go to train #

			self.lock.release()			

			# load parameter #
			 
			self.local_net.load_state_dict(self.global_net.state_dict())

			self.local_net.eval()

			# test #
			for index, data in enumerate(self.dataloader):

				x =  data['point'].cuda()

				labels = data['label'].cuda()

				pred = self.local_net(x)

				#loss#
				loss = nn.MSELoss()(pred, labels)

				loss = loss.detach().cpu().numpy()

				# full 

				self.queue.put(loss)

				# self.writer.add_scalar('./test/loss',loss, global_counter)

				global_counter +=1

			epoch_counter +=1

			if epoch_counter ==self.max_epoch:

				self.finish.value = 1

				break


class trainWoker(mp.Process):

	def __init__(self, global_net, optimizer, update_flag, lock, max_epoch, dataloader, finish, queue):

		super(trainWoker, self).__init__()

		self.global_net = global_net

		self.update_flag = update_flag

		self.lock = lock

		self.max_epoch = max_epoch

		self.dataloader = dataloader

		self.finish = finish

		self.queue = queue

		# self.writer =  SummaryWriter('./logs/train/')

		self.optimizer = optimizer

	def run(self):

		epoch_counter = 0

		global_counter = 0

		self.global_net.train()

		while True:

			self.lock.acquire()

			update_flag = self.update_flag.value

			self.lock.release()

			if update_flag == 1:

				time.sleep(0.001)

				continue

			# train #
			for index, data in enumerate(self.dataloader):

				x =  data['point'].cuda()

				labels = data['label'].cuda()

				pred = self.global_net(x)

				#loss#
				loss = nn.MSELoss()(pred, labels)

				# updata #
				self.optimizer.zero_grad()

				loss.backward()

				self.optimizer.step()

				loss = loss.detach().cpu().numpy()

				self.queue.put(loss)

				# self.writer.add_scalar('./train/loss',loss, global_counter)

				global_counter +=1

			epoch_counter +=1

			self.lock.acquire()

			self.update_flag.value = 1 # go to train #

			self.lock.release()	

			if epoch_counter ==self.max_epoch:

				self.finish.value = 1

				break






			
			
			










