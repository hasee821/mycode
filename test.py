'''
测试最终分割效果,相关文件保存在outputs下,gt是人工分割,predict是模型分割,后缀b指分割边界部分
'''

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from dataset import Horsedata,mycollatefn
from net import Deeplabv3p
from evaluate import get_boundary,boundary_iou,IoU
from PIL import Image

data_path = './weizmann_horse_db/'
total_num = 327
test_num = int(total_num*0.15)
train_num = total_num-test_num
idx = np.arange(total_num)
np.random.seed(2022)
np.random.shuffle(idx)
train_idx = idx[test_num:]
test_idx = idx[:test_num]
train_loader=DataLoader(
    Horsedata(data_path, train_idx, False),
    batch_size=1,
    shuffle=True,
    collate_fn=mycollatefn
)
test_loader=DataLoader(
    Horsedata(data_path, test_idx, False),
    batch_size=1,
    shuffle=True,
    collate_fn=mycollatefn
)

model=Deeplabv3p()
cudnn.benchmark=True
model=model.cuda()
model.load_state_dict(torch.load('model/bestmodel_0.904_0.757.pth'))           #载入最佳模型
model.eval()

biou = 0
miou = 0
for iter,batch in enumerate(train_loader):                   
    with torch.no_grad():
        imgs,labels = batch
        imgs = imgs.cuda()
        outputs = model(imgs).cpu()
        miou+=IoU(outputs,labels)
        biou+=boundary_iou(outputs,labels)
print("训练集mIoU:%.3f boundary iou:%.3f" %(miou/train_num,biou/train_num))
biou = 0
miou = 0
for iter,batch in enumerate(test_loader):                   
    with torch.no_grad():
        imgs,labels = batch
        imgs = imgs.cuda()
        outputs = model(imgs).cpu()
        miou+=IoU(outputs,labels)
        biou+=boundary_iou(outputs,labels)
print("测试集mIoU:%.3f boundary iou:%.3f" %(miou/test_num,biou/test_num))

#显示并保存部分测试集预测图片
for iter,batch in enumerate(test_loader):                   
    with torch.no_grad():
        imgs,labels = batch
        imgs = imgs.cuda()
        outputs = model(imgs).cpu().squeeze(1)
        outputs_b = get_boundary(outputs)
        labels_b = get_boundary(labels)

        #原始图像
        imgs = imgs*255
        imgs = imgs.transpose(1,2).transpose(2,3).cpu().data.numpy()
        orimg = Image.fromarray(imgs[0].astype('uint8'))
        orimg.show()
        orimg.save('outputs/%d.png'%(iter))

        #预测与真实分割图像
        labels = labels.data.numpy()
        outputs = outputs.data.numpy()
        labels[labels > 0.5] = 255
        outputs[outputs > 0.5] = 255
        A = Image.fromarray(labels[0].astype('uint8'))
        B = Image.fromarray(outputs[0].astype('uint8'))
        A.show()
        B.show()
        A.save('outputs/%d_gt.png'%(iter))
        B.save('outputs/%d_predict.png'%(iter))

        #预测与真实边界图像
        labels_b[labels_b > 0.5] = 255
        outputs_b[outputs_b > 0.5] = 255
        C = Image.fromarray(labels_b[0].astype('uint8'))
        D = Image.fromarray(outputs_b[0].astype('uint8'))
        C.show()
        D.show()
        C.save('outputs/%d_gt_b.png'%(iter))
        D.save('outputs/%d_predict_b.png'%(iter))
    
    if iter >= 1:                                             #展示数量
        break