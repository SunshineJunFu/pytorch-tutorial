#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-28 10:48:49
# @Author  : Jun Fu (fujun@mail.ustc.edu.cn)
# @Version : $Id$

import os
import test
import torch
from collections import namedtuple
import torch.multiprocessing as mp
import torchvision.models as models
from torchvision import transforms
import cv2

transform = transforms.Compose([transforms.ToPILImage(),
					transforms.Resize((224, 224)),
        			transforms.ToTensor()])

class worker(mp.Process):

	def __init__(self,):

		super(worker, self).__init__()

		self.vgg = models.vgg19().cuda()

		self.img = transform(cv2.imread('./coaster2_0001.bmp'))

	def run(self,):

		a = torch.randn(1,4,4).cuda()
		b = torch.zeros(a.size()).cuda()

		f = test.m.get_function('flip')

		Stream = namedtuple('Stream', ['ptr'])
		s = Stream(ptr=torch.cuda.current_stream().cuda_stream)

		f(grid=(1,1,1), block=(1024,1,1), args=[b.data_ptr(), a.data_ptr(), a.size(-1), a.numel()],
 		 stream=s)

		print(a)
		print(b)

		fff = self.vgg(self.img.float().cuda().unsqueeze(0))

		print(fff)


if __name__ == '__main__':

	mp.set_start_method('spawn') 

	t = worker()

	t.start()

	t.join()
