import numpy as np
import torch
from torch import nn
from medpy.metric.binary import assd,dc
from datetime import datetime
import scipy.io as scio
import os.path as osp
import torch.backends.cudnn as cudnn

import cv2


def vis_save(original_img, pred, save_path):
    # blue   = [30,144,255] # aorta
    blue = [252,31, 235]
    # green  = [0,255,0]    # gallbladder
    green = [70,70,70]
    # red    = [255,0,0]    # left kidney
    red = [102,102,156]
    # cyan   = [0,255,255]  # right kidney
    # black = [0,0,0]
    # pink   = [255,0,255]  # liver
    # yellow = [255,255,0]  # pancreas
    yellow = [191,152,155]
    # purple = [128,0,255]  # spleen
    # orange = [255,128,0]  # stomach
    bg = [128,63,127]
    original_img = original_img * 255.0
    original_img = original_img.astype(np.uint8)
    original_img = cv2.cvtColor(original_img,cv2.COLOR_GRAY2BGR)
    original_img = np.full_like(original_img, bg)
    # pred = cv2.cvtColor(pred,cv2.COLOR_GRAY2BGR)
    original_img = np.where(np.tile(pred, [3, 1, 1]).transpose(1, 2,0)==1, np.full_like(original_img, blue  ), original_img)
    original_img = np.where(np.tile(pred, [3, 1, 1]).transpose(1, 2,0)==2, np.full_like(original_img, green ), original_img)
    original_img = np.where(np.tile(pred, [3, 1, 1]).transpose(1, 2,0)==3, np.full_like(original_img, red   ), original_img)
    # original_img = np.where(pred==4, np.full_like(original_img, cyan  ), original_img)
    # original_img = np.where(pred==5, np.full_like(original_img, pink  ), original_img)
    original_img = np.where(np.tile(pred, [3, 1, 1]).transpose(1, 2,0)==4, np.full_like(original_img, yellow), original_img)
    # original_img = np.where(pred==7, np.full_like(original_img, purple), original_img)
    # original_img = np.where(pred==8, np.full_like(original_img, orange), original_img)
    original_img = cv2.cvtColor(original_img,cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path, original_img)

BATCHSIZE     = 32
data_size     = [256, 256, 1]
label_size    = [256, 256, 1]
NUMCLASS      = 5

def _compute_metric(pred,target):

    pred = pred.astype(int)
    target = target.astype(int)
    dice_list  = []
    assd_list  = []
    pred_each_class_number = []
    true_each_class_number = []


    for c in range(1,NUMCLASS):
        y_true    = target.copy()
        test_pred = pred.copy()
        test_pred[test_pred != c] = 0
        test_pred[test_pred == c] = 1
        y_true[y_true != c] = 0
        y_true[y_true == c] = 1
        pred_each_class_number.append(np.sum(test_pred))
        true_each_class_number.append(np.sum(y_true))

    for c in range(1, NUMCLASS):
        test_pred = pred.copy()
        test_pred[test_pred != c] = 0

        test_gt = target.copy()
        test_gt[test_gt != c] = 0

        dice = dc(test_pred, test_gt)

        try:
            assd_metric = assd(test_pred, test_gt)
        except:
            print('assd error')
            assd_metric = 1

        dice_list.append(dice)
        assd_list.append(assd_metric)

    return  np.array(dice_list),np.array(assd_list)

def eval(model,testfile_list,TARGET_MODALITY,pretrained_model_pth):



    dice_mean,dice_std,assd_mean,assd_std = eval_uda(testfile_list,model, pretrained_model_pth,TARGET_MODALITY)

    return dice_mean,dice_std,assd_mean,assd_std

def eval_uda(testfile_list,model,pretrained_model_pth,TARGET_MODALITY):

    interp = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

    img_mean   = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    if not osp.exists(pretrained_model_pth):
        print('')
    print('Evaluating model {}'.format(pretrained_model_pth))
    load_checkpoint_for_evaluation(model,pretrained_model_pth)

    dice_list = []
    assd_list = []
    for idx_file, fid in enumerate(testfile_list):
        _npz_dict = np.load(fid)
        data      = _npz_dict['arr_0']
        label     = _npz_dict['arr_1']

        if True:
            data = np.flip(data, axis=0)
            data = np.flip(data, axis=1)
            label = np.flip(label, axis=0)
            label = np.flip(label, axis=1)

        tmp_pred = np.zeros(label.shape)
        frame_list = [kk for kk in range(data.shape[2])]
        pred_start_time = datetime.now()

        for ii in range(int(np.floor(data.shape[2] // BATCHSIZE))):
            data_batch = np.zeros([BATCHSIZE, 3, 256, 256])
            for idx, jj in enumerate(frame_list[ii * BATCHSIZE: (ii + 1) * BATCHSIZE]):
                item_data = data[..., jj]

                if TARGET_MODALITY == 'CT':
                    item_data = np.subtract(
                        np.multiply(np.divide(np.subtract(item_data, -2.8), np.subtract(3.2, -2.8)), 2.0),
                        1)  # {-2.8, 3.2} need to be changed according to the metadata statistics
                elif TARGET_MODALITY == 'MR':
                    item_data = np.subtract(
                        np.multiply(np.divide(np.subtract(item_data, -1.8), np.subtract(4.4, -1.8)), 2.0),
                        1)  # {-1.8, 4.4} need to be changed according to the metadata statistics
                item_data = np.expand_dims(item_data, -1)
                item_data = np.tile(item_data, [1, 1, 3])
                item_data = (item_data + 1) * 127.5
                item_data = item_data[:, :, ::-1].copy()  # change to BGR
                item_data -= img_mean
                item_data = np.transpose(item_data, [2, 0, 1])
                data_batch[idx, ...] = item_data

            imgs = torch.from_numpy(data_batch).cuda().float()
            with torch.no_grad():
                cla_feas_src,pred_b_aux, pred_b_main = model(imgs)

                pred_b_main = interp(pred_b_main)
                pred_b_main = torch.argmax(pred_b_main, dim=1)
                pred_b_main = pred_b_main.cpu().data.numpy()
            for idx, jj in enumerate(frame_list[ii * BATCHSIZE: (ii + 1) * BATCHSIZE]):
                tmp_pred[..., jj] = pred_b_main[idx, ...].copy()

        pred_end_time = datetime.now()
        pred_spend_time = (pred_end_time-pred_start_time).seconds
        print('pred spend time is {} seconds'.format(pred_spend_time))

        for i in range(data.shape[2]):
            original_img = data[..., i]
            predict = tmp_pred[..., i]
            save_path = f"../mr2ct/pred_mr/{fid[31:38]}_{i}.png"
            vis_save(original_img, predict, save_path)



        label = label.astype(int)
        metric_start_time      = datetime.now()
        dice, assd             = _compute_metric(tmp_pred,label)
        metric_end_time        = datetime.now()
        metric_spend_time      = (metric_end_time-metric_start_time).seconds
        print('metric spend time is {} seconds'.format(metric_spend_time))

        dice_list.append(dice)
        assd_list.append(assd)

    dice_arr = np.vstack(dice_list) #N_CT * N_Class
    assd_arr = np.vstack(assd_list) #N_CT * N_Class

    dice_arr  = 100 * dice_arr.transpose()  #N_Class * N_CT
    dice_mean = np.mean(dice_arr, axis=1) #N_Class
    dice_std  = np.std(dice_arr, axis=1) #N_Class

    print('dice arr is {}'.format(dice_arr.shape))
    print('Dice:')
    print('AA :%.2f(%.1f)' % (dice_mean[3], dice_std[3]))
    print('LAC:%.2f(%.1f)' % (dice_mean[1], dice_std[1]))
    print('LVC:%.2f(%.1f)' % (dice_mean[2], dice_std[2]))
    print('Myo:%.2f(%.1f)' % (dice_mean[0], dice_std[0]))
    print('Mean:%.2f' % np.mean(dice_mean))

    assd_arr  = assd_arr.transpose() #N_Class * N_CT
    assd_mean = np.mean(assd_arr, axis=1)
    assd_std  = np.std(assd_arr, axis=1)

    print('ASSD:')
    print('AA :%.2f(%.1f)' % (assd_mean[3], assd_std[3]))
    print('LAC:%.2f(%.1f)' % (assd_mean[1], assd_std[1]))
    print('LVC:%.2f(%.1f)' % (assd_mean[2], assd_std[2]))
    print('Myo:%.2f(%.1f)' % (assd_mean[0], assd_std[0]))
    print('Mean:%.2f' % np.mean(assd_mean))

    return dice_mean,dice_std,assd_mean,assd_std

def load_checkpoint_for_evaluation(model, checkpoint):
    saved_state_dict = torch.load(checkpoint,map_location='cpu')
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True
