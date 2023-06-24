from model.DRNet_V8 import DRNet
from flask import Flask, request 
import io
from PIL import Image  
import torch  
import numpy as np
from torchvision import transforms
from utils.aftertreatment import StrLabelConverter
from utils.fileoperation import get_figure

class ResizeAndNormalize(object):
    def __init__(self, size, interpolation=	Image.Resampling.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        size = self.size
        imgW, imgH = size
        # 等比例放大或缩小图片
        scale = img.size[1] * 1.0 / imgH
        w = img.size[0] / scale
        w = int(w)
        img = img.resize((w, imgH), self.interpolation)
        w, h = img.size
        # 图片宽度小于需要的宽度，则在右方补充255，否则，直接resize
        if w <= imgW:
            newImage = np.zeros((imgH, imgW), dtype='uint8')
            newImage[:] = 255
            newImage[:, :w] = np.array(img)
            img = newImage
        else:
            img = img.resize((imgW, imgH), self.interpolation)

        # 转换成Tensor
        img = self.toTensor(img)
        # 让图像分布在(-1，1)之间
        img.sub_(0.5).div_(0.5)
        return img


#初始化Flask app
app = Flask(__name__)# 创建app，__name__表示指向程序所在的包
model=None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# #返回结果
# with open('./result.txt', 'r')as f:
#     result=eval(f.read())

#加载模型
def load_model():
    """Load the pre-trained model, you can use your model just as easily.

    """
    global model
    model = DRNet(32, 1, 11, 256).to(device)
    model.eval()
        
#数据预处理
def preprocess_image(image):
    """Do image preprocessing before prediction on any data.

    :param image:       original image
    :param target_size: target image size
    :return:
                        preprocessed image
    """
    image = Image.open(image).convert('L')
    transform = ResizeAndNormalize((280, 32))
    return transform(image).unsqueeze(0).to(device)

#启动预测服务
@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
        # Read the image in PIL format
    image = request.files["image"].read()
    image = Image.open(io.BytesIO(image))
    # Preprocess the image and prepare it for classification.
    image = preprocess_image( Image.open(image).convert('L'))

    # Classify the input image and then initialize the list of predictions to return to the client.
    chinese = get_figure('./config//figure.txt')
    converter = StrLabelConverter(chinese)
    preds = model(image)
    y_hat = torch.nn.functional.softmax(preds, 2).argmax(2).view(preds.size(0), -1)
    y_hat = torch.transpose(y_hat, 1, 0)
    results = [converter.decode(i, torch.IntTensor([y_hat.size(1)])) for i in y_hat]
    return results[0]


if __name__ == '__main__':
    print("loading Pytorch model and Flask starting server....")
    load_model()
    app.run()  # 运行app

