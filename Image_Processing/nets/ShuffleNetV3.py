import torch
import torch.nn as nn
import torch.nn.functional as F
from .ECA import eca_layer

def channel_shuffle( x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]

class bottleblock(nn.Module):
    def __init__(self,in_channel, out_channel, mid_channel, stride):
        super(bottleblock, self).__init__()
        self.midchannel=mid_channel
        output=out_channel-in_channel
        self.stride=stride

        self.pointwise_conv1=nn.Sequential(nn.Conv2d(in_channels=in_channel,out_channels=mid_channel,kernel_size=1,stride=1,bias=False),
                                           nn.BatchNorm2d(mid_channel),
                                           nn.ReLU(inplace=True))
        self.depth_conv=nn.Sequential(nn.Conv2d(in_channels=mid_channel,out_channels=mid_channel,kernel_size=3,padding=1,stride=stride,groups=mid_channel,bias=False),
                                      nn.BatchNorm2d(mid_channel))
        self.pointwise_conv2=nn.Sequential(nn.Conv2d(in_channels=mid_channel,out_channels=output,kernel_size=1,stride=1,bias=False),
                                           nn.BatchNorm2d(output),
                                           nn.ReLU(inplace=True))
        if stride==2:
            
            self.shortcut=nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3 ,padding=1, stride=stride, groups=in_channel,bias=False),
                                        nn.BatchNorm2d(in_channel),
                                        nn.Conv2d(in_channels=in_channel,out_channels=in_channel,kernel_size=1,stride=1,bias=False),
                                        nn.BatchNorm2d(in_channel),
                                        nn.ReLU(inplace=True))
        else:
            self.shortcut=nn.Sequential()
            
    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]
    
    def forward(self,x):
        if self.stride==2:
            residual=self.shortcut(x)
            x=self.pointwise_conv1(x)
            x=self.depth_conv(x)
            x=self.pointwise_conv2(x)
            return torch.cat((residual,x),dim=1)
        elif self.stride==1:
            x1,x2=self.channel_shuffle(x)
            residual=self.shortcut(x2)
            x1=self.pointwise_conv1(x1)
            x1=self.depth_conv(x1)
            x1=self.pointwise_conv2(x1)
            return torch.cat((residual,x1),dim=1)

class shufflenet_V2(nn.Module):
    def __init__(self,num_class, size, width):
        """size表示模型大小"""
        super(shufflenet_V2, self).__init__()
        self.num_class=num_class
        self.inchannel=int(128*width)
        if size==0.5:
            stage_dict={'bolck_num':[4,8,4],
                         'outchannel':[48,96,192],
                        'last_conv':1024,
                         'size':size}
        elif size==1:
            stage_dict = {'bolck_num': [4, 8, 4],
                          'outchannel': [int(256*width), int(512*width), int(1024*width)],
                          #'last_conv': 1024,
                          'size': size}
        elif size==1.5:
            stage_dict = {'bolck_num': [4, 8, 4],
                               'outchannel': [176, 352, 704],
                          'last_conv': 1024,
                               'size':size}
        elif size==2:
            stage_dict = {'bolck_num': [4, 8, 4],
                               'outchannel': [244, 488, 976],
                          'last_conv': 2048,
                               'size':size}

        block_num=stage_dict['bolck_num']
        outchannel=stage_dict['outchannel']
        # last_conv=stage_dict['last_conv']
        self.initial=nn.Sequential(nn.Conv2d(kernel_size=3, padding=1, in_channels=3, out_channels=int(128*width), stride=2),
                                   nn.BatchNorm2d(int(128*width)),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self.make_layer(block_num[0], outchannel[0]) 
        self.layer2 = self.make_layer(block_num[1], outchannel[1])
        self.layer3 = self.make_layer(block_num[2], outchannel[2])
        #self.last_conv=nn.Conv2d(in_channels=outchannel[2],out_channels=last_conv,stride=1,kernel_size=1,bias=False)

        # self.pool=nn.AdaptiveAvgPool2d(1)
        # self.fc=nn.Linear(last_conv,num_class)
        self.eca_attention=eca_layer()
        
    def make_layer(self,block_num, outchannel):
        layer_list=[]
        for i in range(block_num):
            if i==0:
                stride=2
                layer_list.append(bottleblock(self.inchannel, outchannel, outchannel//2, stride=stride))
                self.inchannel=outchannel
            else:
                stride=1
                layer_list.append(bottleblock(self.inchannel//2, outchannel, outchannel//2, stride=stride))
        return nn.Sequential(*layer_list)
    def forward(self,x):
        x=self.initial(x)
        p1=x
        
        x=self.layer1(x)
        p2=self.eca_attention(x)
        x=self.layer2(x)
        p3=self.eca_attention(x)
        x=self.layer3(x)
        p4=self.eca_attention(x)
       
        # x=self.layer1(x)
        # p2=x
        # x=self.layer2(x)
        # p3=x
        # x=self.layer3(x)
        # p4=x
       
        return p1, p2, p3, p4





from ptflops import get_model_complexity_info
from torchsummary import summary
if __name__ == '__main__':
    with torch.cuda.device(0):
        net = shufflenet_V2(1, 1)
        macs, params = get_model_complexity_info(net, (3, 640, 640), as_strings=True,
                                            print_per_layer_stat=True, verbose=True) 
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


    



        