# !/usr/bin/env python
# encoding: utf-8
"""         
  @Author: Yongheng Sun          
  @Contact: 3304925266@qq.com          
  @Software: PyCharm    
  @Project: MPSCL
  @File: read_png.py          
  @Time: 2022/5/28 22:34                   
"""
import cv2


path = '../ct2mr/pred_ct/ct_1003_96.png'
path2 = '../mr2ct/label/ct_1003_94.png'

path3 = '../mr2ct/pred_mr/mr_1007_101_my.png'

a = cv2.imread(path)
aa = a.transpose(2,0,1)
# path2 = '../figure/ct_1014_116.png'
a2 = cv2.imread(path2)
aa2 = a2.transpose(2,0,1)