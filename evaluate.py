#损失与评价指标
import torch.nn as nn
import numpy as np
import cv2

def IoU(predict,mask):                           #均交并比计算
    predict = predict > 0.5
    mask = mask > 0.5
    n = predict.size(0)
    predict = predict.view(n,-1)
    mask = mask.view(n,-1)

    intersection1 = (predict & mask).sum(1)                #前景交并比
    union1 = (predict | mask).sum(1)
    intersection0 = (~predict & ~mask).sum(1)              #背景交并比
    union0 = (~predict | ~mask).sum(1)
    iou = (intersection1 / union1 + intersection0 / union0)*0.5       #均交并比
    return iou.sum()


def get_boundary(mask,dilation_ratio=0.02):   #通过腐蚀获取实例边界，dilation_ratio腐蚀率与边界大小有关
    mask = mask.data.numpy()
    mask = mask > 0.5
    mask = mask.astype('uint8')

    b,h,w = mask.shape
    new_mask = np.zeros([b,h+2,w+2])
    for i in range(b):
        new_mask[i] = cv2.copyMakeBorder(mask[i], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)  # 扩展边框
    r = np.sqrt(h**2+w**2) #图像对角线长
    dilation = round(dilation_ratio*r) #腐蚀程度最小为1
    if dilation<1:
        dilation = 1

    #腐蚀边界,得到腐蚀后图像,相减得到边界图像
    erode_mask = np.zeros([b,h,w])
    kernel = np.ones((3,3),dtype=np.uint8)
    for i in range(b):
        tmp = cv2.erode(src=new_mask[i],kernel=kernel,iterations=dilation)
        erode_mask[i] = tmp[1:h+1,1:w+1]
    
    return mask-erode_mask

def boundary_iou(predict,mask):        #计算边界交并比
    pre_boundary = get_boundary(predict.squeeze(1))
    mask_boundary = get_boundary(mask)
    b,h,w = mask_boundary.shape

    biou = 0
    for i in range(b):
        intersection = ((pre_boundary[i]*mask_boundary[i])>0).sum()
        union = ((pre_boundary[i]+mask_boundary[i])>0).sum()
        biou += intersection/union
    return biou


def Myloss(predict,mask,alpha=0.5,gammma=2):   #两部分组成，一部分为交叉熵，一部分为Dice损失
    n = predict.size(0)
    smooth = 1e-5
    temp_predict = predict.view(n,-1)
    temp_mask = mask.view(n,-1)
    
    bce_loss = nn.BCELoss()(temp_predict,temp_mask)
    
    intersection = temp_predict*temp_mask
    dice = (2.*intersection.sum(1)+smooth)/(temp_predict.sum(1)+temp_mask.sum(1)+smooth)
    dice_loss = 1 - dice.mean()
    return bce_loss + dice_loss
