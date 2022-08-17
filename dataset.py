import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

def mycollatefn(batch): #array to tensor
    imgs=[]
    masks=[]
    for img,mask in batch:
        imgs.append(img)
        masks.append(mask)
    imgs = torch.from_numpy(np.array(imgs)).type(torch.FloatTensor)
    masks = torch.from_numpy(np.array(masks)).type(torch.FloatTensor)
    return imgs,masks

class Horsedata(Dataset):
    def __init__(self,path,idx,train):
        #path马数据集目录，idx训练集测试集对应图片编号,train训练模式/测试模式
        self.image_size=(224,224)
        self.path=path
        self.train = train
        self.imgs=list(np.array(list(sorted(os.listdir(os.path.join(path, "horse")))))[idx])
        self.masks=list(np.array(list(sorted(os.listdir(os.path.join(path, "mask")))))[idx])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,i):
        #读取图像数据
        img_path=os.path.join(self.path,'horse',self.imgs[i])
        mask_path=os.path.join(self.path,'mask',self.imgs[i])
        img=Image.open(img_path)
        mask=Image.open(mask_path)
        #数据增强
        img,mask = self.get_random_data(img,mask)
        #图像归一化
        img = np.transpose(np.array(img,np.float64),[2,0,1])/255.0
        return img,mask
    
    def rand(self,a=0,b=1):
        return np.random.rand()*(b-a)+a

    def get_random_data(self,img,mask):
        iw,ih = img.size
        w,h = self.image_size
        jitter =0.3
        scale = min(w/iw,h/ih)

        #测试模式
        if self.train == False:              
            #按小边等比例放缩
            nw = int(iw*scale)
            nh = int(ih*scale)
            img = img.resize((nw,nh),Image.BICUBIC)
            new_img = Image.new('RGB',[w,h],(0,0,0))
            new_img.paste(img,((w-nw)//2, (h-nh)//2))      #将放缩后图片至于固定大小图片中央

            mask = mask.resize((nw,nh),Image.NEAREST)    #标签用最近邻插值
            new_mask = Image.new('L', [w, h], (0))       #缩放后未填充部分为背景
            new_mask.paste(mask, ((w-nw)//2, (h-nh)//2))
            return new_img, np.array(new_mask)
        
        #训练模式
        #长宽比随机扭曲，随机尺度缩放(0.6,1)
        new_sc = iw/ih*self.rand(1-jitter,1+jitter)/self.rand(1-jitter,1+jitter)
        new_scale = self.rand(scale*0.6,scale)   
        if new_sc <1:       #按大边缩放
            nh=int(new_scale*h)
            nw=int(nh*new_sc)
        else:
            nw=int(new_scale*w)
            nh=int(nw/new_sc)
        img = img.resize((nw,nh),Image.BICUBIC)        
        mask = mask.resize((nw,nh),Image.NEAREST)

        #水平翻转
        if self.rand()<0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        #图像随机偏移
        dx = int(self.rand(0,w-nw))
        dy = int(self.rand(0,h-nh))
        new_img = Image.new('RGB',[w,h],(0,0,0))
        new_img.paste(img,(dx,dy))
        new_mask = Image.new('L', [w, h], (0))
        new_mask.paste(mask,(dx,dy))
        img = new_img
        mask = np.array(new_mask)

        #随机5*5高斯模糊
        img_data = np.array(img,np.uint8)
        if self.rand()<0.25:
            img_data = cv2.GaussianBlur(img_data,(5,5),0)
        
        #在HSV空间随机色域变化
        h, s, v   = cv2.split(cv2.cvtColor(img_data, cv2.COLOR_RGB2HSV))
        dtype = img_data.dtype
        r = np.random.uniform(-1, 1, 3) * [0.1, 0.7, 0.3] + 1
        x = np.arange(0, 256, dtype=r.dtype)
        lut_h = ((x * r[0]) % 180).astype(dtype)
        lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_v = np.clip(x * r[2], 0, 255).astype(dtype)
        img_data = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
        img_data = cv2.cvtColor(img_data, cv2.COLOR_HSV2RGB)
        return img_data,mask