from  yolo import YoloBody  
import torch
from ptflops import get_model_complexity_info
from torchsummary import summary
from ShuffleNetV2 import shufflenet

with torch.cuda.device(0):
    net = YoloBody(1, 's')
    #net2 = shufflenet(1, 0.5)
    macs, params = get_model_complexity_info(net, (3, 640, 640), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    #summary(net2,  (3, 640, 640))