import torch, os
from .utils.pretreatment import ResizeAndNormalize
from PIL import Image
import torch.nn as nn
from .utils.aftertreatment import StrLabelConverter
from .utils.fileoperation import get_figure
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed_all(seed)  #并行gpu0
#数字识别
#def digital_rec(file_path, model):
def digital_rec(image, model):
    setup_seed(1)
    with torch.no_grad():
        chinese = get_figure(os.path.join(os.getcwd(), 'Image_Processing\\config\\figure.txt'))
        converter = StrLabelConverter(chinese)
        #crnn=model.to(device)
        #crnn.load_state_dict(torch.load(weigths))
        # 读取灰度图的图片
        #image = Image.open(file_path).convert('L')
        # 对图片进行resize和归一化操作
        transform = ResizeAndNormalize((280, 32))
        image = transform(image).unsqueeze(0).to(device)
        #loader = DataLoader(dataset=image_path, batch_size=1)
        preds = model(image)
        #preds_size = torch.Tensor([preds.size(0)] * batch_size)
        y_hat = nn.functional.softmax(preds, 2).argmax(2).view(preds.size(0), -1)
        y_hat = torch.transpose(y_hat, 1, 0)
        y_hat = [converter.decode(i, torch.IntTensor([y_hat.size(1)])) for i in y_hat]
        rec_result=y_hat[0]
        # A= list(y_hat[0]) # 转化
        # A.insert(-3, '.') # 加小数点
        # # if(len(A)>5):
        # #     del A[0]
        # # else:
        # #     A.insert(-5, '0')
        # rec_result=''.join(A) # 转化回来
        return rec_result

# rec_weights='./weights/drnet_v3.pt'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# rec_model=DRNet(32, 1, 11, 256).to(device)
# rec_model.load_state_dict(torch.load(rec_weights))
# image_path=os.path.join('./img_crop/', ('./data_image/test.jpg').split("/")[-1])
# rec_result= digital_rec(image_path,  rec_model)
# (rec_result)