# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 13:16:32 2017

@author: shriv
"""


import numpy as np

a = np.random.rand(5,3,2,4)

print (str(a.shape))

b = a.reshape(5,-1).T

print(str(b.shape))
