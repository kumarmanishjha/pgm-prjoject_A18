# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 20:44:57 2018

@author: Gabriel Hsu
"""

# http://pytorch.org/
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import pickle
import random
import time

  
path_real_data = 'Fixed_results/'
path_generated_data = 'Random_results/'

if not os.path.exists(path_real_data):
    raise RuntimeError('Invalid path real samples: %s' % path_real_data)
if not os.path.exists(path_generated_data):
    raise RuntimeError('Invalid path generated samples: %s' % path_generated_data)
    
# CPU or GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    use_cuda = True
    print('using cuda !')
else:
    raise RuntimeError('GPU not avaialble !')
    
print(os.getcwd())
#os.chdir('..') 
#print(os.getcwd())
print(os.listdir())

# start the computation of FID
#python D:/CVAE/pgm-prjoject_A18/CVAE/pytorch-fid/fid_score.py  D:/CVAE/pgm-prjoject_A18/CVAE/Fixed_results/  D:/CVAE/pgm-prjoject_A18/CVAE/Random_results/ --gpu 0