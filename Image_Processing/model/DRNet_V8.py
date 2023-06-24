import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .Coordinate_Attention import CA_Block
from .Self_Attention import SelfAttention


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        """
        :param nIn: 输入层神经元个数
        :param nHidden: 隐藏层神经元个数
        :param nOut: 输出层神经元个数
        """
        super(BidirectionalLSTM, self).__init__()
        # 双向LSTM
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        # 两个方向的隐藏层单元频在一起，所以nHidden*2
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        # T:时间序列  b:batch_size   h:隐藏层神经元
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        #print(output.shape)
        return output

# class BidirectionalLSTM_V2(nn.Module):
    
#     def __init__(self, nIn, nHidden, nOut):
#         """
#         :param nIn: 输入层神经元个数
#         :param nHidden: 隐藏层神经元个数
#         :param nOut: 输出层神经元个数
#         """
#         super(BidirectionalLSTM_V2, self).__init__()
#         # 双向LSTM
#         self.rnn=nn.GRU(nIn, nHidden, bidirectional=True)
#         # 两个方向的隐藏层单元频在一起，所以nHidden*2
#         self.embedding = nn.Linear(nHidden * 2, nOut)

#     def forward(self, input):
#         recurrent, _ = self.rnn(input)
#         # T:时间序列  b:batch_size   h:隐藏层神经元
#         T, b, h = recurrent.size()
#         t_rec = recurrent.view(T * b, h)
#         output = self.embedding(t_rec)
#         output = output.view(T, b, -1)
#         #print(output.shape)
#         return output
    
    
# class AttentionLSTM(nn.Module):
#     def __init__(self, nIn, nHidden, nOut):
#         super(AttentionLSTM, self).__init__()
#         self.hidden_size = nHidden
#         self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        
#         self.attention = nn.Linear(nHidden*2, nHidden*2)        
        
#         self.embedding = nn.Linear(nHidden * 2, nOut)
        
#     def forward(self, input):
#         # LSTM编码
#         output, (hidden, cell) = self.rnn(input)
#         # 计算注意力权重
#         attention_weights = torch.softmax(self.attention(output), dim=1)
#         # 计算注意力向量
#         output = attention_weights * output
#         output = self.embedding(output)
#         return output

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


#线性瓶颈和反向残差结构
# class Block(nn.Module):
#     '''expand + depthwise + pointwise'''
#     def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
#         super(Block, self).__init__()
#         self.stride = stride
#         self.se = semodule

#         self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn1 = nn.BatchNorm2d(expand_size)
#         self.nolinear1 = nolinear
#         self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
#         self.bn2 = nn.BatchNorm2d(expand_size)
#         self.nolinear2 = nolinear
#         self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_size)

#         self.shortcut = nn.Sequential()
#         if stride == 1 and in_size != out_size:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(out_size),
#             )

#     def forward(self, x):
#         #print(x.shape)
#         out = self.nolinear1(self.bn1(self.conv1(x)))
#         out = self.nolinear2(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         # print(out.shape)
#         if self.se != None:
#             out = self.se(out)
#             # print(out.shape)
#         out = out + self.shortcut(x) if self.stride==1 else out
#         return out


class Block_V2(nn.Module):
    def __init__(self,kernel_size, in_size, expand_size, out_size, nolinear, module, stride):
        super(Block_V2, self).__init__()
        self.stride=stride
        self.Attention_Module=module
        
        self.conv1=nn.Sequential(
            nn.Conv2d(in_size, expand_size, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(expand_size,  eps=1e-6),
            nolinear
        )
        
        self.conv2=nn.Sequential(
            nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_size,  eps=1e-6),
            nn.MaxPool2d(kernel_size=1, stride=1)
        )
       
        self.conv3=nn.Sequential(
            nn.Conv2d(out_size, out_size, kernel_size, stride, padding=kernel_size//2,  groups=out_size, bias=False),
            nn.BatchNorm2d(out_size, eps=1e-6),
            nolinear
        )
        
        self.shortcut = nn.Sequential()
        #if stride == 1 and in_size != out_size:
        if stride == 1: #and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )
        
    def forward(self, x):
        
        out = self.conv1(x)
        out =self.conv2(out)
        out=self.conv3(out)
        # print(out.shape)
        if self.Attention_Module != None:
            out = self.Attention_Module(out)
            # print(out.shape)
        # out = out + self.shortcut(x) if self.stride==1 else out
        out = out + self.shortcut(x) if self.stride==1 else out
        return out
        

class DRNet(nn.Module):
    def __init__(self,  imgH, nc, nclass, nh):
        """
        :param imgH: 图片高度
        :param nc: 图片通道数
        :param nclass: 类别个数
        :param nh: RNN中隐藏层神经元个数
        """

        super(DRNet, self).__init__()
        assert imgH % 16 == 0, '图片高度必须是16的倍数，建议32'
        
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=nc, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size= 2,stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(8),                       
        )
        # self.blok_0=RestNet18()
        # self.blok_1=nn.Sequential(
        #     Block_V2(3, 8, 32, 16, nn.ReLU(inplace=True), SeModule(16), 1),
        #     #torch.Size([32, 16, 16, 140])
        #     Block_V2(3, 16, 64, 32, nn.ReLU(inplace=True), None, 1),
        #     #torch.Size([32, 32, 16, 140])
        #     Block_V2(3, 32, 72, 64, hswish(),  SeModule(64), (2,1)),
        #     #torch.Size([32, 64, 8, 140])
        #     Block_V2(3, 64, 88, 64, nn.ReLU(),   None,1), #CA_Block(64, 8, 140),
        #     #torch.Size([3, 64, 8, 140])
        #     Block_V2(3, 64, 96, 64, hswish(),  SeModule(64), (2,1)),
        #     #torch.Size([32, 64, 4, 140])
        #     Block_V2(3, 64, 144, 128, nn.ReLU(),  None, 1), #CA_Block(128, 4, 140),
        #     #torch.Size([32, 128, 4, 140])
        #     Block_V2(5, 128, 240, 256, hswish(),  SeModule(256), (2,1)),
        #     #torch.Size([32, 256, 2, 140])
        #     Block_V2(5, 256, 288, 256, nn.ReLU(),  None,1),  #CA_Block(256, 2, 140), 
        #     #torch.Size([32, 256, 2, 140])
        #     Block_V2(5, 256, 576, 256, hswish(),  SeModule(256),  2),
        #     #torch.Size([32, 256, 1, 70])
        #     Block_V2(5, 256, 576, 256, nn.ReLU(),  None,1),  #CA_Block(256, 2,140), 
        #     #torch.Size([32, 256, 1, 70])
        #     nn.Dropout(0.5)
        # )

        self.blok_2=nn.Sequential(
            Block_V2(3, 8, 32, 16, nn.ReLU(inplace=True), CA_Block(16, 16, 140 ), 1),
            #torch.Size([32, 16, 16, 140])
            Block_V2(3, 16, 64, 32, nn.ReLU(inplace=True), None, 1),
            #torch.Size([32, 32, 16, 140])
            Block_V2(3, 32, 72, 64, hswish(),  CA_Block(64, 8, 140), (2,1)),
            #torch.Size([32, 64, 8, 140])
            Block_V2(3, 64, 88, 64, hswish(),   None,1), #CA_Block(64, 8, 140),
            #torch.Size([3, 64, 8, 140])
            Block_V2(3, 64, 96, 64, hswish(),  CA_Block(64, 4, 140), (2,1)),
            #torch.Size([32, 64, 4, 140])
            Block_V2(3, 64, 144, 128, hswish(),  None, 1), #CA_Block(128, 4, 140),
            #torch.Size([32, 128, 4, 140])
            Block_V2(5, 128, 240, 256, hswish(),  CA_Block(256, 2, 140), (2,1)),
            #torch.Size([32, 256, 2, 140])
            Block_V2(5, 256, 288, 256, hswish(),  None,1),  #CA_Block(256, 2, 140), 
            #torch.Size([32, 256, 2, 140])
            Block_V2(5, 256, 576, 256, hswish(),  CA_Block(256, 1, 70), 2),
            #torch.Size([32, 256, 1, 70])
            Block_V2(5, 256, 576, 256, hswish(),  None,1),  #CA_Block(256, 2,140), 
            #torch.Size([32, 256, 1, 70])
            nn.Dropout(0.5)
        )
        
        
        # self.blok_3=nn.Sequential(
        #     Block_V2(3, 8, 32, 16, nn.ReLU(inplace=True), CBAM(16), 1),
        #     #torch.Size([32, 16, 16, 140])
        #     Block_V2(3, 16, 64, 32, nn.ReLU(inplace=True), None, 1),
        #     #torch.Size([32, 32, 16, 140])
        #     Block_V2(3, 32, 72, 64, hswish(),  CBAM(64), (2,1)),
        #     #torch.Size([32, 64, 8, 140])
        #     Block_V2(3, 64, 88, 64, nn.ReLU(),   None,1), #CA_Block(64, 8, 140),
        #     #torch.Size([3, 64, 8, 140])
        #     Block_V2(3, 64, 96, 64, hswish(),  CBAM(64), (2,1)),
        #     #torch.Size([32, 64, 4, 140])
        #     Block_V2(3, 64, 144, 128, nn.ReLU(),  None, 1), #CA_Block(128, 4, 140),
        #     #torch.Size([32, 128, 4, 140])
        #     Block_V2(5, 128, 240, 256, hswish(),  CBAM(256), (2,1)),
        #     #torch.Size([32, 256, 2, 140])
        #     Block_V2(5, 256, 288, 256, nn.ReLU(),  None,1),  #CA_Block(256, 2, 140), 
        #     #torch.Size([32, 256, 2, 140])
        #     Block_V2(5, 256, 576, 256, hswish(),  CBAM(256),  2),
        #     #torch.Size([32, 256, 1, 70])
        #     Block_V2(5, 256, 576, 256, nn.ReLU(),  None,1),  #CA_Block(256, 2,140), 
        #     #torch.Size([32, 256, 1, 70])
        #     nn.Dropout(0.5)
        # )
        
        # self.blok_4=nn.Sequential(
        #     Block_V2(3, 8, 32, 16, nn.ReLU(inplace=True), GAM_Attention(16, 16), 1),
        #     #torch.Size([32, 16, 16, 140])
        #     Block_V2(3, 16, 64, 32, nn.ReLU(inplace=True), None, 1),
        #     #torch.Size([32, 32, 16, 140])
        #     Block_V2(3, 32, 72, 64, hswish(),  GAM_Attention(64, 64), (2,1)),
        #     #torch.Size([32, 64, 8, 140])
        #     Block_V2(3, 64, 88, 64, hswish(),   None,1), #GAM(64, 8, 140),
        #     #torch.Size([3, 64, 8, 140])
        #     Block_V2(3, 64, 96, 64, hswish(),  GAM_Attention(64, 64), (2,1)),
        #     #torch.Size([32, 64, 4, 140])
        #     Block_V2(3, 64, 144, 128, hswish(),  None, 1), #GAM(128, 4, 140),
        #     #torch.Size([32, 128, 4, 140])
        #     Block_V2(5, 128, 240, 256, hswish(),  GAM_Attention(256, 256), (2,1)),
        #     #torch.Size([32, 256, 2, 140])
        #     Block_V2(5, 256, 288, 256, hswish(),  None,1),  #GAM(256, 2, 140), 
        #     #torch.Size([32, 256, 2, 140])
        #     Block_V2(5, 256, 576, 256, hswish(),  GAM_Attention(256, 256), 2),
        #     #torch.Size([32, 256, 1, 70])
        #     Block_V2(5, 256, 576, 256, nn.ReLU(),  None,1),  #GAM(256, 2,140), 
        #     #torch.Size([32, 256, 1, 70])
        #     nn.Dropout(0.2)
        # )
        
        # self.blok_5=nn.Sequential(
        #     Block_V2(3, 8, 32, 16, nn.ReLU(inplace=True), Att(16), 1),
        #     #torch.Size([32, 16, 16, 140])
        #     Block_V2(3, 16, 64, 32, nn.ReLU(inplace=True), None, 1),
        #     #torch.Size([32, 32, 16, 140])
        #     Block_V2(3, 32, 72, 64, hswish(),  Att(64), (2,1)),
        #     #torch.Size([32, 64, 8, 140])
        #     Block_V2(3, 64, 88, 64, nn.ReLU(),   None,1), #CA_Block(64, 8, 140),
        #     #torch.Size([3, 64, 8, 140])
        #     Block_V2(3, 64, 96, 64, hswish(),  Att(64), (2,1)),
        #     #torch.Size([32, 64, 4, 140])
        #     Block_V2(3, 64, 144, 128, nn.ReLU(),  None, 1), #CA_Block(128, 4, 140),
        #     #torch.Size([32, 128, 4, 140])
        #     Block_V2(5, 128, 240, 256, hswish(),  Att(256), (2,1)),
        #     #torch.Size([32, 256, 2, 140])
        #     Block_V2(5, 256, 288, 256, nn.ReLU(),  None,1),  #CA_Block(256, 2, 140), 
        #     #torch.Size([32, 256, 2, 140])
        #     Block_V2(5, 256, 576, 256, hswish(),  Att(256),  2),
        #     #torch.Size([32, 256, 1, 70])
        #     Block_V2(5, 256, 576, 256, nn.ReLU(),  None,1),  #CA_Block(256, 2,140), 
        #     #torch.Size([32, 256, 1, 70])
        #     nn.Dropout(0.5)
        # )
        
        # self.blok_6=nn.Sequential(
        #     Block_V2(3, 16, 32, 16, nn.ReLU(inplace=True), CC_module(16), 1),
        #     #torch.Size([32, 16, 16, 140])
        #     Block_V2(3, 16, 64, 32, nn.ReLU(inplace=True), None, 1),
        #     #torch.Size([32, 32, 16, 140])
        #     Block_V2(3, 32, 72, 64, hswish(),  CC_module(64), (2,1)),
        #     #torch.Size([32, 64, 8, 140])
        #     Block_V2(3, 64, 88, 64, nn.ReLU(),   None,1), #CA_Block(64, 8, 140),
        #     #torch.Size([3, 64, 8, 140])
        #     Block_V2(3, 64, 96, 64, hswish(),  CC_module(64), (2,1)),
        #     #torch.Size([32, 64, 4, 140])
        #     Block_V2(3, 64, 144, 128, nn.ReLU(),  None, 1), #CA_Block(128, 4, 140),
        #     #torch.Size([32, 128, 4, 140])
        #     Block_V2(5, 128, 240, 256, hswish(),  CC_module(256), (2,1)),
        #     #torch.Size([32, 256, 2, 140])
        #     Block_V2(5, 256, 288, 256, nn.ReLU(),  None,1),  #CA_Block(256, 2, 140), 
        #     #torch.Size([32, 256, 2, 140])
        #     Block_V2(5, 256, 576, 256, hswish(),  CC_module(256),  2),
        #     #torch.Size([32, 256, 1, 70])
        #     Block_V2(5, 256, 576, 256, nn.ReLU(),  None,1),  #CA_Block(256, 2,140), 
        #     #torch.Size([32, 256, 1, 70])
        #     nn.Dropout(0.5)
        # )
        
        

        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels= 256, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            hswish()
            #nn.ReLU()            
        )
       
        self.init_params()

        self.rnn_V1 = nn.Sequential(
        BidirectionalLSTM(512, nh, nh),
        SelfAttention(num_attention_heads=1, input_size=nh, hidden_size=nh, hidden_dropout_prob=0.5),
        BidirectionalLSTM(nh, nh, nclass),
        )
        
        # self.rnn_V2 = nn.Sequential(
        # BidirectionalLSTM_V2(512, nh, nh),
        # SelfAttention(num_attention_heads=1, input_size=nh, hidden_size=nh, hidden_dropout_prob=0.5),
        # BidirectionalLSTM_V2(nh, nh, nclass),
        # )


    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        #特征提取网络
        # out=self.blok_0(x)  
        out =self.conv1(x)   #[32, 8, 32, 280]
        out=self.blok_2(out)   # [32, 256, 2, 140]
        out =self.conv3(out) #32, 512, 1, 70]
        #序列RNN
        b, c, h, w = out.size() #output: ([32, 512, 1, 70])
        assert h == 1, '图片高度经过卷积之后必须为1'
        out = out.squeeze(2)   #output: ([32, 512, 70)
        out = out.permute(2, 0, 1)  # [w, b, c]
        # out = self.rnn(out)     # [seq * batch * n_classes] [70, 32, 11]
        out=self.rnn_V1(out)
        return out



# from torchsummary import summary
# if __name__ == '__main__':
#     opt = parse_opt()
#     model = DRNet(opt.imgH, opt.nc, opt.nclass, opt.nh)
#     model1=Block_V2(3, 8, 16, 16, nn.GeLU(), SeModule(16), 2)
#     summary(model1,(8, 32, 280))
        # col_width=30,
        # col_names=["kernel_size", "input_size","output_size", "num_params"])