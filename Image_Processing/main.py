import sys
import os
import cv2
import torch
from PIL import Image
from model.config import config_model
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from Ui_Yolox_OCR import Ui_MainWindow  #导入你写的界面类
from yolox import YOLO
from model.DRNet_V8 import DRNet
from digital_rec import digital_rec
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UI_Logic_Window(QtWidgets.QMainWindow):
    def __init__(self, parent =None):
        super(UI_Logic_Window,self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.init_slots()
        self.opt = config_model()
        self.out_floder='./img_crop/'
        self.ui.event_message.setText("系统初始化")
        self.ui.event_message.append("……………………………")
        
    def init_slots(self):
        self.ui.model.clicked.connect(self.model_load)
        self.ui.image.clicked.connect(self.image_load)
        self.ui.detect.clicked.connect(self.image_detect)
        #self.timer_video.timeout.connect(self.show_video_frame)
        
    #模型加载
    def model_load(self):
        #加载分割模型权重
        self.openfile_det_model, _ = QFileDialog.getOpenFileName(self.ui.model, '选择det_weights文件',  './weights/')
        if not self.openfile_det_model:
            det_weights=self.opt.det_model_path
            self.ui.event_message.append('Warning','加载分割权重失败, 将使用默认权重文件')
            self.ui.event_message.append("……………………………")
            self.ui.event_message.append("加载det_weights文件地址为：" +str(self.opt.det_model_path))
            self.ui.event_message.append("……………………………")
        else:
            det_weights = self.openfile_det_model
            self.ui.event_message.append('权重文件加载成功，将使用当前权重')
            self.ui.event_message.append("……………………………")
            self.ui.event_message.append("加载det_weights文件地址为：" +str(self.openfile_det_model))
            self.ui.event_message.append("……………………………")
            
        #加载f分割模型
        self.det_model=YOLO(det_weights)
        self.ui.event_message.append('分割模型加载成功')
        self.ui.event_message.append("……………………………")
        
        #加载识别模型权重
        self.openfile_rec_model, _ = QFileDialog.getOpenFileName(self.ui.model, '选择det_weights文件',  './weights/')
        if not self.openfile_rec_model:
            det_weights=self.opt.rec_model_path
            self.ui.event_message.append('Warning','加载识别权重失败, 将使用默认权重文件')
            self.ui.event_message.append("……………………………")
            self.ui.event_message.append("加载det_weights文件地址为：" +str(self.opt.rec_model_path))
            self.ui.event_message.append("……………………………")
        else:
            rec_weights = self.openfile_rec_model
            self.ui.event_message.append('权重文件加载成功，将使用当前权重')
            self.ui.event_message.append("……………………………")
            self.ui.event_message.append("加载det_weights文件地址为：" +str(self.openfile_rec_model))
            self.ui.event_message.append("……………………………")
            
        #加载识别模型
        self.rec_model=DRNet(32, 1, 11, 256).to(device)
        self.rec_model.load_state_dict(torch.load(rec_weights))
        self.ui.event_message.append('识别模型加载成功')
        self.ui.event_message.append("……………………………")
        
    #图像读取
    def image_load(self):
        try:
            self.image_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开图片", "./data_image/", "*.jpg;;*.png;;All Files(*)")
        except OSError as reason:
            self.ui.event_message.append('文件打开出错啦！核对路径是否正确'+ str(reason))
            self.ui.event_message.append("……………………………")
        else:
            # 判断图片是否为空
            if not self.image_path:
                self.ui.event_message.append('Warning','图片加载失败, 请重新加载')
                self.ui.event_message.append("……………………………")
            else:
                img = cv2.imread(self.image_path)
                self.origin_image= cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                self.origin_QtImg = QtGui.QImage(self.origin_image.data, self.origin_image.shape[1], self.origin_image.shape[0], QtGui.QImage.Format_RGB32)
                self.ui.origin_image.setPixmap(QtGui.QPixmap.fromImage(self.origin_QtImg))
                self.ui.origin_image.setScaledContents(True) # 设置图像自适应界面大小
                self.ui.event_message.append('图片加载成功')
                self.ui.event_message.append("……………………………")
                
                
    #图像检测与识别        
    def image_detect(self):
        self.ui.event_message.append('开始图片分割')
        self.ui.event_message.append("……………………………")
        image_path=os.path.join(self.out_floder, self.image_path.split("/")[-1])
        self.det_model.detect_image( Image.open(self.image_path), crop=True)
        self.detect_image= cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2BGRA)
        self.detect_QtImg = QtGui.QImage(self.detect_image.data, self.detect_image.shape[1], self.detect_image.shape[0], QtGui.QImage.Format_RGB32)
        self.ui.detect_image.setPixmap(QtGui.QPixmap.fromImage(self.detect_QtImg))
        self.ui.detect_image.setScaledContents(True) # 设置图像自适应界面大小
        self.ui.event_message.append('图片分割完成')
        self.ui.event_message.append("……………………………")
        #detect_image.save(image_path, quality=95, subsampling=0)
        self.ui.event_message.append('开始图片识别')
        self.ui.event_message.append("……………………………")
        rec_result= digital_rec(image_path,  self.rec_model)
        self.ui.result_message.append(self.image_path.split("/")[-1]+'：'+rec_result)
        self.ui.event_message.append('图片识别完成')
        self.ui.event_message.append("……………………………")
        
        
        
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = UI_Logic_Window()
    myWin.show()
    sys.exit(app.exec_())    