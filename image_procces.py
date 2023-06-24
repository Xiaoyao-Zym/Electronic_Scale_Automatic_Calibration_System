from Image_Processing.yolox_V1 import YOLO
from Image_Processing.model.DRNet_V8 import DRNet
from Image_Processing.digital_rec import digital_rec
from PIL import Image
import os, torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


image_path='./Image_Processing/image/'
file=os.listdir(image_path)
rec_weights='./Image_Processing/weights/drnet_v8_01.pt'
rec_model=DRNet(32, 1, 11, 256).to(device)
rec_model.eval()
rec_model.load_state_dict(torch.load(rec_weights))
with open('./1.txt', 'w', encoding='utf-8') as f:
    for line in file:
        rec_image=Image.open(os.path.join(image_path,  line)).convert('L')
        rec= digital_rec(rec_image,  rec_model)
        print(rec)
        f.write(line+' '+rec+'\n')
        

