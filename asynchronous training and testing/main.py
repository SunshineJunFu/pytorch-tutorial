#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-01 14:06:47
# @Author  : Jun Fu (fujun@mail.ustc.edu.cn)
# @Version : $Id$

from Worker import *

from model import *

from dataloader import *

from  tensorboardX import SummaryWriter 

import torch.multiprocessing as mp  


if __name__ == '__main__':

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	train_writer = SummaryWriter('./logs/train/')

	test_writer = SummaryWriter('./logs/test/')

	mp.set_start_method('spawn')

	gnet = regressModel(1,1).cuda()

	gnet.share_memory()

	optimizer = torch.optim.Adam(gnet.parameters())

	# optimizer.share_memory()

	lnet = regressModel(1,1).cuda()

	train_finsh = mp.Value('i',0)

	test_finsh = mp.Value('i',0)

	train_loss_queue = mp.Queue(int(1e6))

	test_loss_queue = mp.Queue(int(1e6))

	updata_flag = mp.Value('i', 0)

	lock = mp.Lock()

	max_epoch = 1000

	simD = simDataLoader(1000,100)

	trainLoader, testLoader = simD.getLoader()

	test_work = testWorker(gnet, lnet, updata_flag, lock, max_epoch,testLoader, train_finsh, train_loss_queue)

	train_work = trainWoker(gnet,optimizer,updata_flag, lock, max_epoch, trainLoader, test_finsh, test_loss_queue)

	test_work.start()

	train_work.start()

	train_counter = 0

	test_counter = 0


	while True:

		if test_finsh.value==1 and train_finsh.value==1:
			break
		else:
			if train_loss_queue.empty():
				pass
			else:
				train_writer.add_scalar('loss', train_loss_queue.get(), train_counter)
				train_counter +=1
			if test_loss_queue.empty():
				pass
			else:
				test_writer.add_scalar('loss', test_loss_queue.get(), test_counter)

				test_counter +=1


	test_work.join()

	train_work.join()




