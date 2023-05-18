#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2021/8/26
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : config.py
# @Software : Python3.6;PyCharm;Windows10 / Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M / 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090
# @Version  : V1.0
# @License  : None
# @Brief    : 配置文件

import os
import matplotlib
import platform
import pandas as pd
import random
import numpy as np

os.environ["OUTDATED_IGNORE"] = "1"  # 忽略OutdatedPackageWarning
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 15)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 200)
np.set_printoptions(threshold=np.inf)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'  # 保存矢量图中的文本在AI中可编辑


if platform.system() == 'Windows':
	# font_family = 'Times New Roman'
	font_family = 'Arial'
else:
	font_family = 'DejaVu Sans'
matplotlib.rcParams["font.family"] = font_family


def setup_seed(seed: int):
	"""
	全局固定随机种子
	:param seed: 随机种子值
	:return: None
	"""
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	np.random.seed(seed)


rs = 2269
setup_seed(rs)


