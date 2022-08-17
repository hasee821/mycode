import torch
import torch.nn as nn
import torch.nn.functional as F

#v2 bneck块
class bneck(nn.Module):
    '''
    采用深度可分离卷积depthwise和pointwise convolution,减少参数量,激活函数采用relu6
    输出输入有相同shape时采用残差连接

    exp_ratio:中间层通道放大倍数
    dilation:膨胀率,大于1为空洞卷积
    '''
    def __init__(self,in_channels,out_channels,stride,exp_ratio,dilation=1):
        super(bneck,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride             #1,hw不变; 2,hw减半
        padding = dilation
        mid_channels = round(in_channels*exp_ratio)

        if exp_ratio == 1:     #中间层不升维
            self.conv = nn.Sequential(
                #depth-wise conv
                nn.Conv2d(mid_channels,mid_channels,3,stride=stride,padding=padding,dilation=dilation,groups=mid_channels,bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU6(inplace=True),
                #point_wise conv
                nn.Conv2d(mid_channels,out_channels,1,stride=1,padding=0,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:                  #升维
            self.conv = nn.Sequential(
                #先升维
                nn.Conv2d(in_channels,mid_channels,1,stride=1,padding=0,bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU6(inplace=True),
                #depth-wise conv
                nn.Conv2d(mid_channels,mid_channels,3,stride=stride,padding=padding,dilation=dilation,groups=mid_channels,bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU6(inplace=True),
                #point_wise conv
                nn.Conv2d(mid_channels,out_channels,1,stride=1,padding=0,bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self,x):
        out = self.conv(x)
        out = (out + x) if self.stride == 1 and self.in_channels == self.out_channels else out
        return out

# deeplab V3+主干网络,下采样率为8
class MobileNetV2(nn.Module):
    '''
    采用mobilenetV2作为deeplabv3+特征提取的主干部分,并将下采样率由32改为8,
    并在深层加入空洞卷积弥补感受野损失.
    两次下采样作为浅层特征,4次下采样作为深层特征
    '''
    def __init__(self):
        super(MobileNetV2,self).__init__()
        bneck_setting = [
            #exp_rotio,out_channel,repeat_num,first_stride,dilation
            [1,16,1,1,1],          
            [6,24,2,2,1],            #两次下采样浅层特征
            [6,32,3,2,1],
            [6,64,4,1,2],
            [6,96,3,1,2],
            [6,160,3,1,4],           
            [6,320,1,1,4]            #3次下采样深层特征
        ]
        self.features = []
        in_channels = 32 

        #first_conv,V2头部
        self.features.append(nn.Sequential(
            nn.Conv2d(3,in_channels,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        ))
        #V2主干特征提取
        for t,c,n,s,d in bneck_setting:
            for i in range(n):
                if i==0 and s==2:
                    if d==4:
                        self.features.append(bneck(in_channels,c,2,t,2))
                    else:
                        self.features.append(bneck(in_channels,c,2,t,1))
                else:
                    self.features.append(bneck(in_channels,c,1,t,d))
                in_channels = c
        self.features = nn.Sequential(*self.features)

    def forward(self,x):
        shallow = self.features[:4](x)
        x =self.features[4:](shallow)    
        return shallow,x

#ASPP空洞卷积金字塔
class ASPP(nn.Module):
    '''
    利用不同膨胀率的空洞卷积提取不同尺度下信息并进行合并,压缩处理深层特征
    rate:基础膨胀率
    '''
    def __init__(self,in_channels,out_channels,rate=1):
        super(ASPP,self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=rate,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=6*rate,dilation=6*rate,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=12*rate,dilation=12*rate,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=18*rate,dilation=18*rate,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_channels*5,out_channels,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch5_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0,bias=True)
        self.branch5_bn = nn.BatchNorm2d(out_channels)
        self.branch5_relu = nn.ReLU(inplace=True)
    
    def forward(self,x):
        [b,c,row,col]=x.size()
        #前4个分支为4个不同卷积大小和膨胀率的空洞卷积
        conv1_1 = self.branch1(x)
        conv3_1 = self.branch2(x)
        conv3_2 = self.branch3(x)
        conv3_3 = self.branch4(x)

        #分支5为全局平均池化+1*1卷积+插值上采样
        branch5 = torch.mean(x,2,True)
        branch5 = torch.mean(branch5,3,True)
        branch5 = self.branch5_conv(branch5)
        branch5 = self.branch5_bn(branch5)
        branch5 = self.branch5_relu(branch5)
        branch5 = F.interpolate(branch5,(row,col),mode='bilinear',align_corners=True)

        #5个分支拼接+1*1卷积整合
        x = torch.cat([conv1_1,conv3_1,conv3_2,conv3_3,branch5],dim=1)
        x = self.conv_cat(x)
        return x

#deeplabv3+网络
class Deeplabv3p(nn.Module):
    '''
    分为encoder和decoder部分
    encoder利用mobilenetv2和aspp提取深层特征
    decoder利用encoder得到的深层特征与浅层特征融合解码输出预测结果
    '''
    def __init__(self,pretrain=True):
        super(Deeplabv3p,self).__init__()
        self.backbone = MobileNetV2()
        if pretrain:  #公开预训练权重地址:http://sceneparsing.csail.mit.edu/model/pretrained_resnet/mobilenet_v2.pth.tar
            self.backbone.load_state_dict(torch.load("./model/mobilenet_v2.pth.tar"),strict=False)
        self.aspp = ASPP(in_channels=320,out_channels=256,rate=1)

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(24,48,1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+256,256,3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256,256,3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256,1,kernel_size=1,stride=1)
    
    def forward(self,x):
        [b,c,row,col]=x.size()
        #encoder部分
        low_feature,x = self.backbone(x)
        x = self.aspp(x)                 

        #decoder部分
        low_feature = self.shortcut_conv(low_feature)
        x = F.interpolate(x,(low_feature.size(2),low_feature.size(3)),mode='bilinear',align_corners=True)
        x = self.cat_conv(torch.cat((x,low_feature),dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x,(row,col),mode='bilinear',align_corners=True)
        x = nn.Sigmoid()(x)
        return x

if __name__ == '__main__':

    model = Deeplabv3p()
    input = torch.randn(2, 3, 16, 16)
    out = model(input)
    print(out)