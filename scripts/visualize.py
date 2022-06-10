# !/usr/bin/env python
# encoding: utf-8
"""         
  @Author: Yongheng Sun          
  @Contact: 3304925266@qq.com          
  @Software: PyCharm    
  @Project: MPSCL
  @File: visualize.py          
  @Time: 2022/5/28 20:33                   
"""
import cv2


def vis_save(original_img, pred, save_path):
    blue   = [30,144,255] # aorta
    green  = [0,255,0]    # gallbladder
    red    = [255,0,0]    # left kidney
    cyan   = [0,255,255]  # right kidney
    pink   = [255,0,255]  # liver
    yellow = [255,255,0]  # pancreas
    purple = [128,0,255]  # spleen
    orange = [255,128,0]  # stomach
    original_img = original_img * 255.0
    original_img = original_img.astype(np.uint8)
    original_img = cv2.cvtColor(original_img,cv2.COLOR_GRAY2BGR)
    pred = cv2.cvtColor(pred,cv2.COLOR_GRAY2BGR)
    original_img = np.where(pred==1, np.full_like(original_img, blue  ), original_img)
    original_img = np.where(pred==2, np.full_like(original_img, green ), original_img)
    original_img = np.where(pred==3, np.full_like(original_img, red   ), original_img)
    original_img = np.where(pred==4, np.full_like(original_img, cyan  ), original_img)
    original_img = np.where(pred==5, np.full_like(original_img, pink  ), original_img)
    original_img = np.where(pred==6, np.full_like(original_img, yellow), original_img)
    original_img = np.where(pred==7, np.full_like(original_img, purple), original_img)
    original_img = np.where(pred==8, np.full_like(original_img, orange), original_img)
    original_img = cv2.cvtColor(original_img,cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path, original_img)