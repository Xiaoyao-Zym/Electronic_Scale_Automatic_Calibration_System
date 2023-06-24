import torch
import torch.nn as nn
from .network_component import BaseConv, Bottleneck
from .ShuffleNetV3 import shufflenet_V2

class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu",):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  
        #--------------------------------------------------#
        #   主干部分的初次卷积
        #--------------------------------------------------#
        self.conv1  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        #--------------------------------------------------#
        #   大的残差边部分的初次卷积
        #--------------------------------------------------#
        self.conv2  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        #-----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        #-----------------------------------------------#
        self.conv3  = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)

        #--------------------------------------------------#
        #   根据循环的次数构建上述Bottleneck残差结构
        #--------------------------------------------------#
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        self.m      = nn.Sequential(*module_list)

    def forward(self, x):
        #-------------------------------#
        #   x_1是主干部分
        #-------------------------------#
        x_1 = self.conv1(x)
        #-------------------------------#
        #   x_2是大的残差边部分
        #-------------------------------#
        x_2 = self.conv2(x)

        #-----------------------------------------------#
        #   主干部分利用残差结构堆叠继续进行特征提取
        #-----------------------------------------------#
        x_1 = self.m(x_1)
        #-----------------------------------------------#
        #   主干部分和大的残差边部分进行堆叠
        #-----------------------------------------------#
        x = torch.cat((x_1, x_2), dim=1)
        #-----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        #-----------------------------------------------#
        return self.conv3(x)

class CSPDarknet(nn.Module):
    def __init__(self, width, out_features=("dark3", "dark4", "dark5"), ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.net = shufflenet_V2(1, 1, width)
        # Conv = DWConv if depthwise else BaseConv

        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        #-----------------------------------------------#
        # base_channels   = int(wid_mul * 64)  # 64
        # base_depth      = max(round(dep_mul * 1), 3)  # 3
        
        # # stage_dict = {'bolck_num': [4, 8, 4],
        # #                     'outchannel': [int(128*wid_mul), int(256*wid_mul) , int(512*wid_mul)],
        # #                     }
        # # block_num=stage_dict['bolck_num']
        # # out_channel=stage_dict['outchannel']
        # # #-----------------------------------------------#
        # # #   利用focus网络结构进行特征提取 
        # # #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        # # #-----------------------------------------------#
        # self.stem = Focus(3, base_channels, ksize=3, act=act)
        # self.stem = Focus(base_depth, base_channels, ksize=3, act=act)
        # #-----------------------------------------------#
        # #   完成卷积之后，320, 320, 64 -> 160, 160, 128
        # #   完成CSPlayer之后，160, 160, 128 -> 160, 160, 128
        # #-----------------------------------------------#
        # self.dark2 = nn.Sequential(
        #     #Conv(base_channels, base_channels * 2, 3, 1, act=act),
        #     # CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
        #     shufflenet_V2(out_channel[0]//2,  block_num[0], out_channel[0])
        # )
        # #-----------------------------------------------#
        # #   完成卷积之后，160, 160, 128 -> 80, 80, 256
        # #   完成CSPlayer之后，80, 80, 256 -> 80, 80, 256
        # #-----------------------------------------------#
        # self.dark3 = nn.Sequential(
        #     # Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
        #     # CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        #     shufflenet_V2(out_channel[1]//2,  block_num[1], out_channel[1])
        # )

        # #-----------------------------------------------#
        # #   完成卷积之后，80, 80, 256 -> 40, 40, 512
        # #   完成CSPlayer之后，40, 40, 512 -> 40, 40, 512
        # #-----------------------------------------------#
        # self.dark4 = nn.Sequential(
        #     # Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
        #     # CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        #     shufflenet_V2(out_channel[2]//2,  block_num[2], out_channel[2])
        # )

        # #-----------------------------------------------#
        # #   完成卷积之后，40, 40, 512 -> 20, 20, 1024
        # #   完成SPP之后，20, 20, 1024 -> 20, 20, 1024
        # #   完成CSPlayer之后，20, 20, 1024 -> 20, 20, 1024
        # #-----------------------------------------------#
        # self.dark5 = nn.Sequential(
        #     Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
        #     SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
        #     CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act),
        # )

    def forward(self, x):
        outputs = {}
        p1, p2, p3, p4=self.net(x)
        #-----------------------------------------------#
        #   stem 的输出为320, 320, 64
        #-----------------------------------------------#
        # x = self.stem(x)
        outputs["stem"] = p1
        #-----------------------------------------------#
        #   dark2的输出为160, 160, 128
        #-----------------------------------------------#
        # x = self.dark2(x)
        # outputs["dark2"] = p2
        #-----------------------------------------------#
        #   dark3的输出为80, 80, 256，是一个有效特征层
        #-----------------------------------------------#
        # x = self.dark3(x)
        outputs["dark3"] = p2
        #-----------------------------------------------#
        #   dark4的输出为40, 40, 512，是一个有效特征层
        #-----------------------------------------------#
        # x = self.dark4(x)
        outputs["dark4"] = p3
        #-----------------------------------------------#
        #   dark5的输出为20, 20, 1024，是一个有效特征层
        #-----------------------------------------------#
        # x = self.dark5(x)
        outputs["dark5"] = p4
        return {k: v for k, v in outputs.items() if k in self.out_features}

if __name__ == '__main__':
    module=CSPDarknet(width=0.5)
    x=torch.rand(1, 3, 640, 640)
    # net = YoloBody(1, 's')
    # out=net(x)
    print(module(x))