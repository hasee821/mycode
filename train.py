import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from dataset import Horsedata,mycollatefn
from net import Deeplabv3p
from evaluate import Myloss,IoU,boundary_iou
import matplotlib.pyplot as plt

def main():
    f_batch_size = 8                       #冻结时batch
    uf_batch_size = 4                      #未冻结时参数大占内存大因此调小batch
    freeze_epoch = 25
    unfreeze_epoch = 50
    data_path = './weizmann_horse_db/'
    total_num = 327
    test_num = int(total_num*0.15)
    train_num = total_num-test_num
    freeze_lr=5e-4
    unfreeze_lr=5e-4
    

    #固定seed划分训练测试集
    idx = np.arange(total_num)
    np.random.seed(2022)
    np.random.shuffle(idx)
    train_idx = idx[test_num:]
    test_idx = idx[:test_num]
    
    
    #GPU加速
    model=Deeplabv3p()
    cudnn.benchmark = True
    model = model.cuda()
    #model.load_state_dict(torch.load("model/freeze_model.pth"))                               #加载历史模型
    


    '''
    在自己电脑上进行实验,调小了batch_size

    主干网络mobilenet采用了预训练数据进行迁移学习,
    故先将冻结主干参数,只训练其他部分,训练30个epoch后
    解冻主干,训练全局,在进行60个epoch,可以更多

    另外由于采用数据增强后,更难拟合,miou上涨缓慢
    故本实验先采用数据增强训练集训练一遍,
    再用原始训练集训练进行微调。最终测试集miou在0.93左右,调得最佳模型miou=0.94,biou=0.757
    '''
    train_loss_list = []
    train_miou_list = []
    train_biou_list = []
    test_miou_list = []
    test_biou_list = []
    
    
    #冻结主干,训练,输入图片大小统一为224*224
    optimizer=torch.optim.Adam(model.parameters(),lr=freeze_lr,weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.94)
    train_loader=DataLoader(
        Horsedata(data_path, train_idx, train = False),                  
        batch_size=f_batch_size,
        shuffle=True,
        collate_fn=mycollatefn
    )
    test_loader=DataLoader(
        Horsedata(data_path, test_idx, train = False),
        batch_size=f_batch_size,
        shuffle=False,
        collate_fn=mycollatefn
    )
    for param in model.backbone.parameters():             #冻结主干
        param.requires_grad = False
    for epoch in range(freeze_epoch):
        model.train()
        loss_sum=0
        miou = 0
        biou = 0
        for iter,batch in enumerate(train_loader):
            imgs,labels = batch
            with torch.no_grad():
                imgs = imgs.cuda()
            optimizer.zero_grad()
            outputs = model(imgs).cpu()
            loss = Myloss(outputs,labels)
            iou = IoU(outputs,labels)
            biou += boundary_iou(outputs,labels)
            loss_sum+=loss.data.numpy()
            miou+=iou
            loss.backward()
            optimizer.step()
        print("epoch:%d loss:%f miou:%f" %(epoch+1,loss_sum,miou/train_num))
        train_loss_list.append(loss_sum/(iter+1))
        train_miou_list.append(miou/train_num)
        train_biou_list.append(biou/train_num)
        lr_scheduler.step()
        #验证
        model.eval()
        miou = 0
        biou = 0
        for iter,batch in enumerate(test_loader):
            with torch.no_grad():
                imgs,labels = batch
                imgs = imgs.cuda()
                outputs = model(imgs).cpu()
                miou+=IoU(outputs,labels)
                biou+=boundary_iou(outputs,labels)
        print("test mIoU:%f" %(miou/test_num))
        test_miou_list.append(miou/test_num)                      
        test_biou_list.append(biou/test_num)  

    #torch.save(model.state_dict(), 'model/freeze_model.pth')
    

    #解冻全局微调
    optimizer=torch.optim.Adam(model.parameters(),lr=unfreeze_lr,weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.94) #学习率调整,随迭代次数下降
    train_loader=DataLoader(
        Horsedata(data_path, train_idx, False),                                     #训练集是否进行数据增强
        batch_size=uf_batch_size,
        shuffle=True,
        collate_fn=mycollatefn
    )
    test_loader=DataLoader(
        Horsedata(data_path, test_idx, False),
        batch_size=uf_batch_size,
        shuffle=False,
        collate_fn=mycollatefn
    )
    for param in model.backbone.parameters():
        param.requires_grad = True
    for epoch in range(freeze_epoch,unfreeze_epoch):
        model.train()
        loss_sum=0
        miou = 0
        biou = 0
        for iter,batch in enumerate(train_loader):
            imgs,labels = batch
            with torch.no_grad():
                imgs = imgs.cuda()
            optimizer.zero_grad()
            outputs = model(imgs).cpu()
            loss = Myloss(outputs,labels)
            iou = IoU(outputs,labels)
            biou +=boundary_iou(outputs,labels)
            loss_sum+=loss.data.numpy()
            miou+=iou
            loss.backward()
            optimizer.step()
        print("epoch:%d loss:%f miou:%f" %(epoch+1,loss_sum,miou/train_num))
        train_loss_list.append(loss_sum/(iter+1))
        train_miou_list.append(miou/train_num)
        train_biou_list.append(biou/train_num)
        lr_scheduler.step()
        model.eval()
        miou = 0
        biou = 0
        for iter,batch in enumerate(test_loader):
            with torch.no_grad():
                imgs,labels = batch
                imgs = imgs.cuda()
                outputs = model(imgs).cpu()
                miou+=IoU(outputs,labels)
                biou+=boundary_iou(outputs,labels)
        print("test mIoU:%f\nbiou:%f" %(miou/test_num,biou/test_num))
        test_miou_list.append(miou/test_num)                      
        test_biou_list.append(biou/test_num)
        if miou/test_num>0.94:
            torch.save(model.state_dict(), 'model/bestmodel_%.3f_%.3f.pth'%(miou/test_num,biou/test_num))

    #torch.save(model.state_dict(), 'model/last_model.pth')
    
    #保存训练测试记录表格
    epoch_list = range(1,unfreeze_epoch+1)
    plt.figure(figsize=(8, 8))
    plt.title('Mean Iou')
    plt.xlabel('EPOCH')
    plt.ylabel('Mean Iou')
    plt.plot(epoch_list, train_miou_list, label='Train')
    plt.plot(epoch_list, test_miou_list, label='Test')
    plt.legend(loc='upper left')
    plt.savefig("./log/mean_iou.png")
    plt.clf()

    plt.figure(figsize=(8, 8))
    plt.title('boundary Iou')
    plt.xlabel('EPOCH')
    plt.ylabel('boundary Iou')
    plt.plot(epoch_list, train_biou_list, label='Train')
    plt.plot(epoch_list, test_biou_list, label='Test')
    plt.legend(loc='upper left')
    plt.savefig("./log/boundary_iou.png")
    plt.clf()

    plt.figure(figsize=(8, 8))
    plt.title('train loss')
    plt.xlabel('EPOCH')
    plt.ylabel('loss')
    plt.plot(epoch_list, train_loss_list, label='Train')
    plt.legend(loc='upper right')
    plt.savefig("./log/train_loss.png")
    plt.clf()

if __name__ == '__main__':
    main()