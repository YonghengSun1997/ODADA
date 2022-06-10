# !/usr/bin/env python
# encoding: utf-8
"""         
  @Author: Yongheng Sun          
  @Contact: 3304925266@qq.com          
  @Software: PyCharm    
  @Project: MPSCL
  @File: npy2jpg.py          
  @Time: 2022/5/28 20:46                   
"""
import numpy as np
import scipy.misc
from PIL import Image
import os
from matplotlib import pyplot as plt

path = '../data/data_np/test_ct'
dir = os.listdir(path)
for a in dir:
    stem , suffix =os.path.splitext(a)
    imgs_test = np.load(str(path)+str(a),allow_pickle=True) #读入.npy文件
    data = imgs_test.item()
    #print(data)
    cam = data['cam']
    img = tensor_to_np(cam)
    im = Image.fromarray(img)

    im.save('保存路径/'+stem+'.jpg')
