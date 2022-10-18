import random
import numpy as np
import torch


def set_rand_seed(seed=1):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # 保证每次返回得的卷积算法是确定的


set_rand_seed(seed=1)
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import math
# import visdom
import torch.utils.data as Data
import argparse
import numpy as np
import sys
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.autograd import Variable

from distutils.version import LooseVersion
from Datasets.2018 import 2018_dataset
from utils.transform import 2018_transform, 2018_transform_320, 2018_transform_newdata

from Models.model_mia1201 import UNet_DA
from Models.model_mia1201_resnet import UNet_DA_resnet
# import lib

from utils.dice_loss import get_soft_label, val_dice, SoftDiceLoss
from utils.dice_loss import Intersection_over_Union
from utils_new.dice_loss_github import SoftDiceLoss_git, CrossentropyND
from metrics import jaccard_index, f1_score, LogNLLLoss, classwise_f1
from utils_tf import JointTransform2D, ImageToImage2D, Image2D

from utils.evaluation import AverageMeter
from utils.binary import assd, dc, jc, precision, sensitivity, specificity, F1, ACC
from torch.optim import lr_scheduler
from data import PrepareDataset, Rescale, ToTensor, Normalize
from time import *
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch import nn

from itertools import cycle
import surface_distance as surfdist
from medpy import metric


criterion = "loss_MedT"  # loss_A-->SoftDiceLoss;  loss_B-->softdice_git;  loss_C-->CE_softdice_git


class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


criterion_tf = LogNLLLoss()
bce = torch.nn.BCELoss()
device = torch.device("cuda")
"""adding label smoothing"""
real_label=1.
fake_label=0.

def train(trainloader_a, trainloader_b, model, criterion, scheduler, optimizer1, optimizer2, args, epoch):
    losses = AverageMeter()

    model.train()
    for step, data in enumerate(zip(trainloader_a, cycle(trainloader_b))):
    
        
        for name, param in model.named_parameters():
            if "feature_extractor" in name:
                param.requires_grad = True

        (x_a, y_a) = data[0]
        (x_b, y_b) = data[1]
        if not (x_a.shape[0] == args.batch_size):
            # print(step)
            continue
        if not (x_b.shape[0] == args.batch_size):
            # print(step)
            continue
        image_a = Variable(x_a.cuda())
        target_a = Variable(y_a.long().squeeze(dim=1).cuda())
        image_b = Variable(x_b.cuda())
        target_b = Variable(y_b.long().squeeze(dim=1).cuda())

        final_a, _, _, _ = model(x_a)

        loss_seg = criterion_tf(final_a, target_a)


        loss = loss_seg
        losses.update(loss.data, image_a.size(0))

        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()


        for name, param in model.named_parameters():
            if "feature_extractor" in name:
                param.requires_grad = False

        (x_a, y_a) = data[0]
        (x_b, y_b) = data[1]
        if not (x_a.shape[0] == args.batch_size):
            continue
        if not (x_b.shape[0] == args.batch_size):
            # print(step)
            continue
        image_a = Variable(x_a.cuda())
        target_a = Variable(y_a.long().squeeze(dim=1).cuda())
        image_b = Variable(x_b.cuda())
        target_b = Variable(y_b.long().squeeze(dim=1).cuda())


        final_a, loss_orthogonal_a, prob_di_a, prob_ds_a = model(x_a, 2)
        final_b, loss_orthogonal_b, prob_di_b, prob_ds_b = model(x_b, 2)

        prob_di_source = torch.full((args.batch_size,), real_label).cuda()
        prob_di_target = torch.full((args.batch_size,), fake_label).cuda()
        prob_ds_source = torch.full((args.batch_size,), real_label).cuda()         
        prob_ds_target = torch.full((args.batch_size,), fake_label).cuda()

        loss_seg = criterion_tf(final_a, target_a)
        loss_class = bce(prob_di_a, prob_di_source) + bce(prob_di_b, prob_di_target) + bce(prob_ds_a, prob_ds_source) + bce(prob_ds_b, prob_ds_target)

        loss = loss_seg + loss_orthogonal_a.mean() + loss_orthogonal_b.mean() + loss_class * 0.1
        losses.update(loss.data, image_a.size(0))

        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        

        if step % (math.ceil(float(len(trainloader_a.dataset)) / args.batch_size)) == 0:
            print('current lr: {} | Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {losses.avg:.6f}'.format(
                optimizer.state_dict()['param_groups'][0]['lr'],
                epoch, step * len(image_a), len(trainloader_a.dataset),
                       100. * step / len(trainloader_a), losses=losses))

    print('The average loss:{losses.avg:.4f}'.format(losses=losses))
    return losses.avg


def valid(valid_loader_a, valid_loader_b, model, criterion, optimizer, args, epoch, best_score):
    Jaccard = []
    dc = []

    model.eval()
    for step, (t, k) in tqdm(enumerate(valid_loader_a), total=len(valid_loader_a)):
        image = t.float().cuda()
        target = k.float().cuda()

        output, _, _, _ = model(image)

        # output = model(image)                                             # model output
        output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
        output_dis_test = output_dis.permute(0, 2, 3, 1).float()
        target_test = target.permute(0, 2, 3, 1).float()
        b_Jaccard = jc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())
        b_dc = dc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())
        Jaccard.append(b_Jaccard)
        dc.append(b_dc)

    Jaccard_mean = np.average(Jaccard)
    dc_mean = np.average(dc)
    net_score_a = Jaccard_mean + dc_mean

    Jaccard = []
    dc = []

    model.eval()
    for step, (t, k) in tqdm(enumerate(valid_loader_b), total=len(valid_loader_b)):
        image = t.float().cuda()
        target = k.float().cuda()

        output, _, _, _ = model(image)
        # model output
        output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
        output_dis_test = output_dis.permute(0, 2, 3, 1).float()
        target_test = target.permute(0, 2, 3, 1).float()
        b_Jaccard = jc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())
        b_dc = dc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())
        Jaccard.append(b_Jaccard)
        dc.append(b_dc)

    Jaccard_mean = np.average(Jaccard)
    dc_mean = np.average(dc)
    net_score_b = Jaccard_mean + dc_mean


    net_score = net_score_a

    print('The Dice score: {dice: .4f}; '
          'The JC score: {jc: .4f}'.format(
        dice=dc_mean, jc=Jaccard_mean))

    if net_score > max(best_score):
        best_score.append(net_score)
        print(best_score)
        modelname = args.ckpt + '/' + 'best_score' + '_' + args.data + '_checkpoint.pth.tar'
        print('the best model will be saved at {}'.format(modelname))
        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
        torch.save(state, modelname)

    return Jaccard_mean, dc_mean, net_score


def test(test_loader_a, test_loader_b, model, num_para, args):
    modelname = args.ckpt + '/' + 'best_score' + '_' + args.data + '_checkpoint.pth.tar'
    if os.path.isfile(modelname):
        print("=> Loading checkpoint '{}'".format(modelname))
        checkpoint = torch.load(modelname)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Loaded saved the best model at (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> No checkpoint found at '{}'".format(modelname))

    dice = []
    iou = []
    # assd = []
    acc = []
    sensitive = []
    specificy = []
    precision = []
    f1_score = []
    Jaccard_M = []
    Jaccard_N = []
    Jaccard = []
    dc = []
    infer_time = []
    mask_gt_all = []
    mask_pred_all = []
    asd_2D_all = []
    model.eval()
    for step, (img, lab) in tqdm(enumerate(test_loader_a), total=len(test_loader_a)):
        image = img.float().cuda()
        target = lab.float().cuda()  # [batch, 1, 224, 320]

        begin_time = time()

        output, _, _, _ = model(image)

        end_time = time()
        pred_time = end_time - begin_time
        infer_time.append(pred_time)

        output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
        output_dis_test = output_dis.permute(0, 2, 3, 1).float()
        target_test = target.permute(0, 2, 3, 1).float()
        output_soft = get_soft_label(output_dis, 2)
        target_soft = get_soft_label(target, 2)

        mask_gt = target.squeeze().cpu().numpy().astype(bool)
        mask_pred = output_dis.squeeze().cpu().detach().numpy().astype(bool)
        mask_gt_all.append(mask_gt)
        mask_pred_all.append(mask_pred)
        if np.max(mask_pred) != False:
            asd_2D = metric.binary.asd(mask_gt, mask_pred)
            asd_2D_all.append(asd_2D)

        label_arr = np.squeeze(target_soft.cpu().numpy()).astype(np.uint8)
        output_arr = np.squeeze(output_soft.cpu().byte().numpy()).astype(np.uint8)

        b_dice = val_dice(output_soft, target_soft, 2)  # the dice accuracy
        b_iou = Intersection_over_Union(output_dis_test, target_test, 1)  # the iou accuracy
        # b_asd = assd(output_arr[:, :, 1], label_arr[:, :, 1])        # the assd
        b_acc = ACC(output_dis_test.cpu().numpy(), target_test.cpu().numpy())  # the acc
        b_sensitive = sensitivity(output_dis_test.cpu().numpy(), target_test.cpu().numpy())  # the sensitivity
        b_specificy = specificity(output_dis_test.cpu().numpy(), target_test.cpu().numpy())  # the specificity
        b_precision = precision(output_dis_test.cpu().numpy(), target_test.cpu().numpy())  # the precision
        b_f1_score = F1(output_dis_test.cpu().numpy(), target_test.cpu().numpy())  # the F1
        b_Jaccard_m = jc(output_arr[:, :, 1], label_arr[:, :, 1])  # the jc melanoma
        b_Jaccard_n = jc(output_arr[:, :, 0], label_arr[:, :, 0])  # the jc no-melanoma
        b_Jaccard = jc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())
        b_dc = dc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())

        dice_np = b_dice.data.cpu().numpy()
        iou_np = b_iou.data.cpu().numpy()

        dice.append(dice_np)
        iou.append(iou_np)
        # assd.append(b_asd)
        acc.append(b_acc)
        sensitive.append(b_sensitive)
        specificy.append(b_specificy)
        precision.append(b_precision)
        f1_score.append(b_f1_score)
        Jaccard_M.append(b_Jaccard_m)
        Jaccard_N.append(b_Jaccard_n)
        Jaccard.append(b_Jaccard)
        dc.append(b_dc)

    mask_gt = np.array(mask_gt_all).astype(bool)
    mask_pred = np.array(mask_pred_all).astype(bool)
    surface_distances = surfdist.compute_surface_distances(mask_gt, mask_pred,
                                                           spacing_mm=(1.0, 1.0, 1.0))  # need to know apscing_mm
    avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
    hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95)
    surface_overlap = surfdist.compute_surface_overlap_at_tolerance(surface_distances, 1)
    surface_dice = surfdist.compute_surface_dice_at_tolerance(surface_distances, 1)
    volume_dice = surfdist.compute_dice_coefficient(mask_gt, mask_pred)
    print('avg_surf_dist_3D:', avg_surf_dist)
    print('hd_dist_95_3D:', hd_dist_95)
    print('surface_overlap_3D:', surface_overlap)
    print('surface_dice_3D:', surface_dice)
    print('volume_dice_3D:', volume_dice)
    ads_2D_mean = np.average(asd_2D_all)
    ads_2D_std = np.std(asd_2D_all)
    print('The mean asd_2D: {ads_2D_mean: .4f}; The ads_2D std: {ads_2D_std: .4f}'.format(
        ads_2D_mean=ads_2D_mean, ads_2D_std=ads_2D_std))
    all_time = np.sum(infer_time)
    dice_mean = np.average(dice)
    dice_std = np.std(dice)

    iou_mean = np.average(iou)
    iou_std = np.std(iou)

    # assd_mean = np.average(assd)
    # assd_std = np.std(assd)

    acc_mean = np.average(acc)
    acc_std = np.std(acc)

    sensitive_mean = np.average(sensitive)
    sensitive_std = np.std(sensitive)

    specificy_mean = np.average(specificy)
    specificy_std = np.std(specificy)

    precision_mean = np.average(precision)
    precision_std = np.std(precision)

    f1_score_mean = np.average(f1_score)
    if1_score_std = np.std(f1_score)

    Jaccard_M_mean = np.average(Jaccard_M)
    Jaccard_M_std = np.std(Jaccard_M)

    Jaccard_N_mean = np.average(Jaccard_N)
    Jaccard_N_std = np.std(Jaccard_N)

    Jaccard_mean = np.average(Jaccard)
    Jaccard_std = np.std(Jaccard)

    dc_mean = np.average(dc)
    dc_std = np.std(dc)

    print('The  mean dice: {dice_mean: .4f}; The  dice std: {dice_std: .4f}'.format(
        dice_mean=dice_mean, dice_std=dice_std))
    print('The  mean IoU: {iou_mean: .4f}; The  IoU std: {iou_std: .4f}'.format(
        iou_mean=iou_mean, iou_std=iou_std))
    print('The  mean ACC: {acc_mean: .4f}; The  ACC std: {acc_std: .4f}'.format(
        acc_mean=acc_mean, acc_std=acc_std))
    print(
        'The  mean sensitive: {sensitive_mean: .4f}; The  sensitive std: {sensitive_std: .4f}'.format(
            sensitive_mean=sensitive_mean, sensitive_std=sensitive_std))
    print(
        'The  mean specificy: {specificy_mean: .4f}; The  specificy std: {specificy_std: .4f}'.format(
            specificy_mean=specificy_mean, specificy_std=specificy_std))
    print(
        'The  mean precision: {precision_mean: .4f}; The  precision std: {precision_std: .4f}'.format(
            precision_mean=precision_mean, precision_std=precision_std))
    print('The  mean f1_score: {f1_score_mean: .4f}; The  f1_score std: {if1_score_std: .4f}'.format(
        f1_score_mean=f1_score_mean, if1_score_std=if1_score_std))
    print(
        'The  mean Jaccard_M: {Jaccard_M_mean: .4f}; The  Jaccard_M std: {Jaccard_M_std: .4f}'.format(
            Jaccard_M_mean=Jaccard_M_mean, Jaccard_M_std=Jaccard_M_std))
    print(
        'The  mean Jaccard_N: {Jaccard_N_mean: .4f}; The  Jaccard_N std: {Jaccard_N_std: .4f}'.format(
            Jaccard_N_mean=Jaccard_N_mean, Jaccard_N_std=Jaccard_N_std))
    print('The  mean Jaccard: {Jaccard_mean: .4f}; The  Jaccard std: {Jaccard_std: .4f}'.format(
        Jaccard_mean=Jaccard_mean, Jaccard_std=Jaccard_std))
    print('The  mean dc: {dc_mean: .4f}; The  dc std: {dc_std: .4f}'.format(
        dc_mean=dc_mean, dc_std=dc_std))
    print('The inference time: {time: .4f}'.format(time=all_time))
    print("Number of trainable parameters {0} in Model {1}".format(num_para, args.id))

    dice = []
    iou = []
    acc = []
    sensitive = []
    specificy = []
    precision = []
    f1_score = []
    Jaccard_M = []
    Jaccard_N = []
    Jaccard = []
    dc = []
    infer_time = []
    mask_gt_all = []
    mask_pred_all = []
    asd_2D_all = []
    model.eval()
    for step, (img, lab) in tqdm(enumerate(test_loader_b), total=len(test_loader_b)):
        image = img.float().cuda()
        target = lab.float().cuda()  # [batch, 1, 224, 320]

        begin_time = time()

        output, _, _, _ = model(image)

        end_time = time()
        pred_time = end_time - begin_time
        infer_time.append(pred_time)

        output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
        output_dis_test = output_dis.permute(0, 2, 3, 1).float()
        target_test = target.permute(0, 2, 3, 1).float()
        output_soft = get_soft_label(output_dis, 2)
        target_soft = get_soft_label(target, 2)

        mask_gt = target.squeeze().cpu().numpy().astype(bool)
        mask_pred = output_dis.squeeze().cpu().detach().numpy().astype(bool)
        mask_gt_all.append(mask_gt)
        mask_pred_all.append(mask_pred)
        if np.max(mask_pred) != False:
            asd_2D = metric.binary.asd(mask_gt, mask_pred)
            asd_2D_all.append(asd_2D)

        label_arr = np.squeeze(target_soft.cpu().numpy()).astype(np.uint8)
        output_arr = np.squeeze(output_soft.cpu().byte().numpy()).astype(np.uint8)

        b_dice = val_dice(output_soft, target_soft, 2)  # the dice accuracy
        b_iou = Intersection_over_Union(output_dis_test, target_test, 1)  # the iou accuracy
        b_acc = ACC(output_dis_test.cpu().numpy(), target_test.cpu().numpy())  # the acc
        b_sensitive = sensitivity(output_dis_test.cpu().numpy(), target_test.cpu().numpy())  # the sensitivity
        b_specificy = specificity(output_dis_test.cpu().numpy(), target_test.cpu().numpy())  # the specificity
        b_precision = precision(output_dis_test.cpu().numpy(), target_test.cpu().numpy())  # the precision
        b_f1_score = F1(output_dis_test.cpu().numpy(), target_test.cpu().numpy())  # the F1
        b_Jaccard_m = jc(output_arr[:, :, 1], label_arr[:, :, 1])  # the jc melanoma
        b_Jaccard_n = jc(output_arr[:, :, 0], label_arr[:, :, 0])  # the jc no-melanoma
        b_Jaccard = jc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())
        b_dc = dc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())

        dice_np = b_dice.data.cpu().numpy()
        iou_np = b_iou.data.cpu().numpy()

        dice.append(dice_np)
        iou.append(iou_np)
        acc.append(b_acc)
        sensitive.append(b_sensitive)
        specificy.append(b_specificy)
        precision.append(b_precision)
        f1_score.append(b_f1_score)
        Jaccard_M.append(b_Jaccard_m)
        Jaccard_N.append(b_Jaccard_n)
        Jaccard.append(b_Jaccard)
        dc.append(b_dc)
    mask_gt = np.array(mask_gt_all).astype(bool)
    mask_pred = np.array(mask_pred_all).astype(bool)
    surface_distances = surfdist.compute_surface_distances(mask_gt, mask_pred,
                                                           spacing_mm=(1.0, 1.0, 1.0))  # need to know apscing_mm
    avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
    hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95)
    surface_overlap = surfdist.compute_surface_overlap_at_tolerance(surface_distances, 1)
    surface_dice = surfdist.compute_surface_dice_at_tolerance(surface_distances, 1)
    volume_dice = surfdist.compute_dice_coefficient(mask_gt, mask_pred)
    print('avg_surf_dist_3D:', avg_surf_dist)
    print('hd_dist_95_3D:', hd_dist_95)
    print('surface_overlap_3D:', surface_overlap)
    print('surface_dice_3D:', surface_dice)
    print('volume_dice_3D:', volume_dice)
    ads_2D_mean = np.average(asd_2D_all)
    ads_2D_std = np.std(asd_2D_all)
    print('The mean asd_2D: {ads_2D_mean: .4f}; The ads_2D std: {ads_2D_std: .4f}'.format(
        ads_2D_mean=ads_2D_mean, ads_2D_std=ads_2D_std))
    all_time = np.sum(infer_time)
    dice_mean = np.average(dice)
    dice_std = np.std(dice)

    iou_mean = np.average(iou)
    iou_std = np.std(iou)

    acc_mean = np.average(acc)
    acc_std = np.std(acc)

    sensitive_mean = np.average(sensitive)
    sensitive_std = np.std(sensitive)

    specificy_mean = np.average(specificy)
    specificy_std = np.std(specificy)

    precision_mean = np.average(precision)
    precision_std = np.std(precision)

    f1_score_mean = np.average(f1_score)
    if1_score_std = np.std(f1_score)

    Jaccard_M_mean = np.average(Jaccard_M)
    Jaccard_M_std = np.std(Jaccard_M)

    Jaccard_N_mean = np.average(Jaccard_N)
    Jaccard_N_std = np.std(Jaccard_N)

    Jaccard_mean = np.average(Jaccard)
    Jaccard_std = np.std(Jaccard)

    dc_mean = np.average(dc)
    dc_std = np.std(dc)

    print('The  mean dice: {dice_mean: .4f}; The  dice std: {dice_std: .4f}'.format(
        dice_mean=dice_mean, dice_std=dice_std))
    print('The  mean IoU: {iou_mean: .4f}; The  IoU std: {iou_std: .4f}'.format(
        iou_mean=iou_mean, iou_std=iou_std))
    print('The  mean ACC: {acc_mean: .4f}; The  ACC std: {acc_std: .4f}'.format(
        acc_mean=acc_mean, acc_std=acc_std))
    print(
        'The  mean sensitive: {sensitive_mean: .4f}; The  sensitive std: {sensitive_std: .4f}'.format(
            sensitive_mean=sensitive_mean, sensitive_std=sensitive_std))
    print(
        'The  mean specificy: {specificy_mean: .4f}; The  specificy std: {specificy_std: .4f}'.format(
            specificy_mean=specificy_mean, specificy_std=specificy_std))
    print(
        'The  mean precision: {precision_mean: .4f}; The  precision std: {precision_std: .4f}'.format(
            precision_mean=precision_mean, precision_std=precision_std))
    print('The  mean f1_score: {f1_score_mean: .4f}; The  f1_score std: {if1_score_std: .4f}'.format(
        f1_score_mean=f1_score_mean, if1_score_std=if1_score_std))
    print(
        'The  mean Jaccard_M: {Jaccard_M_mean: .4f}; The  Jaccard_M std: {Jaccard_M_std: .4f}'.format(
            Jaccard_M_mean=Jaccard_M_mean, Jaccard_M_std=Jaccard_M_std))
    print(
        'The  mean Jaccard_N: {Jaccard_N_mean: .4f}; The  Jaccard_N std: {Jaccard_N_std: .4f}'.format(
            Jaccard_N_mean=Jaccard_N_mean, Jaccard_N_std=Jaccard_N_std))
    print('The  mean Jaccard: {Jaccard_mean: .4f}; The  Jaccard std: {Jaccard_std: .4f}'.format(
        Jaccard_mean=Jaccard_mean, Jaccard_std=Jaccard_std))
    print('The  mean dc: {dc_mean: .4f}; The  dc std: {dc_std: .4f}'.format(
        dc_mean=dc_mean, dc_std=dc_std))
    print('The inference time: {time: .4f}'.format(time=all_time))
    print("Number of trainable parameters {0} in Model {1}".format(num_para, args.id))



def main(args):
    best_score = [0]
    start_epoch = args.start_epoch
    print('loading the {0},{1},{2} dataset ...'.format('train', 'test', 'test'))

    trainloader_a = torch.utils.data.DataLoader(
        PrepareDataset(args.dataset_a, train=True,
                       transform=transforms.Compose([
                           Normalize(),
                           Rescale(args.out_size),
                           ToTensor()
                       ]),
                       target_transform=transforms.Compose([
                           Normalize(),
                           Rescale(args.out_size),
                           ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=True)
    validloader_a = torch.utils.data.DataLoader(
        PrepareDataset(args.dataset_a, train=False,
                       transform=transforms.Compose([
                           Normalize(),
                           Rescale(args.out_size),
                           ToTensor()
                       ]),
                       target_transform=transforms.Compose([
                           Normalize(),
                           Rescale(args.out_size),
                           ToTensor()
                       ])),
        batch_size=1, shuffle=True)
    testloader_a = torch.utils.data.DataLoader(
        PrepareDataset(args.dataset_a, train=False,
                       transform=transforms.Compose([
                           Normalize(),
                           Rescale(args.out_size),
                           ToTensor()
                       ]),
                       target_transform=transforms.Compose([
                           Normalize(),
                           Rescale(args.out_size),
                           ToTensor()
                       ])),
        batch_size=1, shuffle=True)

    trainloader_b = torch.utils.data.DataLoader(
        PrepareDataset(args.dataset_b, train=True,
                       transform=transforms.Compose([
                           Normalize(),
                           Rescale(args.out_size),
                           ToTensor()
                       ]),
                       target_transform=transforms.Compose([
                           Normalize(),
                           Rescale(args.out_size),
                           ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=True)
    validloader_b = torch.utils.data.DataLoader(
        PrepareDataset(args.dataset_b, train=False,
                       transform=transforms.Compose([
                           Normalize(),
                           Rescale(args.out_size),
                           ToTensor()
                       ]),
                       target_transform=transforms.Compose([
                           Normalize(),
                           Rescale(args.out_size),
                           ToTensor()
                       ])),
        batch_size=1, shuffle=True)
    testloader_b = torch.utils.data.DataLoader(
        PrepareDataset(args.dataset_b, train=False,
                       transform=transforms.Compose([
                           Normalize(),
                           Rescale(args.out_size),
                           ToTensor()
                       ]),
                       target_transform=transforms.Compose([
                           Normalize(),
                           Rescale(args.out_size),
                           ToTensor()
                       ])),
        batch_size=1, shuffle=True)


    print('Loading is done\n')
    args.num_input = 3
    args.num_classes = 2

    model = UNet_DA()
    model = nn.DataParallel(model)
    model = model.cuda()

    print("------------------------------------------")
    print("Network Architecture of Model {}:".format(args.id))
    num_para = 0
    for name, param in model.named_parameters():
        num_mul = 1
        for x in param.size():
            num_mul *= x
        num_para += num_mul
    print(model)
    print("Number of trainable parameters {0} in Model {1}".format(num_para, args.id))
    print("------------------------------------------")

    # Define optimizers and loss function
    optimizer1 = torch.optim.Adam([
        {'params': model.parameters(), 'lr': args.lr_rate, 'weight_decay': args.weight_decay},
    ])
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer1, T_0=10, T_mult=2, eta_min=0.000001)  # lr_3
    for name, param in model.named_parameters():
        if "feature_extractor" in name:
            param.requires_grad = False
    optimizer2 = torch.optim.Adam([
        {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr_rate, 'weight_decay': args.weight_decay},
    ])

    print("Start training ...")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        scheduler.step()
        train_avg_loss = train(trainloader_a, trainloader_b, model, criterion, scheduler, optimizer1, optimizer2, args, epoch)
        Jaccard_mean, dc_mean, net_score = valid(validloader_a, validloader_b, model, criterion, optimizer, args, epoch, best_score)
        if epoch > args.particular_epoch:
            if epoch % args.save_epochs_steps == 0:
                filename = args.ckpt + '/' + str(epoch) + '_' + args.data + '_checkpoint.pth.tar'
                print('the model will be saved at {}'.format(filename))
                state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
                torch.save(state, filename)

    print('Training Done! Start testing')
    if args.data == '2018':
        test(testloader_a, testloader_b, model, num_para, args)
    print('Testing Done!')

if __name__ == '__main__':


    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), 'PyTorch>=0.4.0 is required'
    parser = argparse.ArgumentParser(description='ODADA')

    parser.add_argument('--id', default="UNet_DA",
                        help='Unet...')  # 模型名字
    parser.add_argument('--id_load', default="UNet_individual",
                        help='Unet...')
    # Path related arguments
    parser.add_argument('--root_path', default='/home/gpu1/10T_disk/ddw/skin_lesion_0114/CA_net/data/2018_npy_all',
                        help='root directory of data')
    parser.add_argument('--ckpt',
                        default='./1109/prostate_8bc_lr1e-3_UNetODADA_3step121_GC_C',
                        help='folder to output checkpoints')  # 模型保存的文件夹

    parser.add_argument('--ckpt_a',
                        default='./0406_prostate_BIDMC_lossmedt_img224_400epoch_16bc_lr-4_UNet_individual_cuda7',
                        help='folder to output checkpoints')
    parser.add_argument('--ckpt_b',
                        default='./0406_prostate_HK_lossmedt_img224_400epoch_16bc_lr-4_UNet_individual_cuda6',
                        help='folder to output checkpoints')
    parser.add_argument('--data', default='2018', help='choose the dataset')  # 训练数据

    parser.add_argument('--dataset_a', default='BIDMC', help='choose the dataset')
    parser.add_argument('--dataset_b', default='HK', help='choose the dataset')
    parser.add_argument('--out_size', default=(224, 224), help='the output image size')
    parser.add_argument('--val_folder', default='folder3', type=str,
                        help='which cross validation folder')  # 五折训练

    # optimization related arguments
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 12)')
    parser.add_argument('--lr_rate', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.001)')  # 初始学习率
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--num_input', default=3, type=int,
                        help='number of input image for each patient')
    parser.add_argument('--weight_decay', default=1e-8, type=float, help='weights regularizer')
    parser.add_argument('--particular_epoch', default=30, type=int,
                        help='after this number, we will save models more frequently')
    parser.add_argument('--save_epochs_steps', default=200, type=int,
                        help='frequency to save models after a particular number of epochs')
    parser.add_argument('--resume', default='',
                        help='the checkpoint that resumes from')

    args = parser.parse_args()

    args.ckpt = os.path.join(args.ckpt, args.data, args.val_folder, args.id + "_{}".format(criterion))  # 模型保存地址
    args.ckpt_a = os.path.join(args.ckpt_a, args.data, args.val_folder,
                               args.id_load + "_{}".format(criterion))  # 模型保存地址
    args.ckpt_b = os.path.join(args.ckpt_b, args.data, args.val_folder,
                               args.id_load + "_{}".format(criterion))  # 模型保存地址
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)
    logfile = os.path.join(args.ckpt, '{}_{}_{}.txt'.format(args.val_folder, args.id, criterion))  # 训练日志保存地址
    sys.stdout = Logger(logfile)

    print('Models are saved at %s' % (args.ckpt))
    print("Input arguments:")
    for key, value in vars(args).items():
        print("{:16} {}".format(key, value))

    # if args.start_epoch > 1:
    args.resume_a = args.ckpt_a + '/' + 'best_score' + '_' + args.data + '_checkpoint.pth.tar'
    args.resume_b = args.ckpt_b + '/' + 'best_score' + '_' + args.data + '_checkpoint.pth.tar'
    main(args)
