#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-01 13:31:03
# @Author  : Jun Fu (fujun@mail.ustc.edu.cn)
# @Version : $Id$



import torch.nn as nn 


class regressModel(nn.Module):

	def __init__(self, input_dim, output_dim):

		super(regressModel, self).__init__()

		self.input = nn.Sequential(

			nn.Linear(input_dim, 1024),

			nn.LayerNorm(1024),

			nn.ReLU()

			)

		self.feature = nn.Sequential(

			nn.Linear(1024, 256),

			nn.LayerNorm(256),

			nn.ReLU(),

			nn.Dropout(0.5),

			nn.Linear(256, 128),

			nn.LayerNorm(128),

			nn.ReLU(),

			nn.Dropout(0.5)

			)

		self.ouput = nn.Sequential(

			nn.Linear(128, output_dim)
			)

	def forward(self, x):

		x = self.input(x)

		x = self.feature(x)

		x = self.ouput(x)

		return x


