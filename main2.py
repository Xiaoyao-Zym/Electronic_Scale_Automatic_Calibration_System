import sys
import os
import typing
from PyQt5.QtCore import QObject

from Camera.MVGigE import *
import shutil

from motion_Control import motion_Control
# from model.config import config_model
from PyQt5 import  QtWidgets
from PyQt5.QtWidgets import QApplication,  QMessageBox
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from Ui_ESACS import Ui_MainWindow #导入你写的界面类
from Camera.SingleGrab import MVCam

class UI_Logic_Window(QtWidgets.QMainWindow): 
    def __init__(self, parent =None):
        super(UI_Logic_Window,self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.data=[]
        self.temp_str=[]
        # self.winid = self.Image_Show.winId()  # 获取label对象的句柄
        self.ui.event_message.setText("系统初始化")
        self.ui.event_message.append("……………………………")
        self.thread = motion_Control()
        self.init_slots()
        
    
    def init_slots(self):
        self.ui.Camera_Connect.clicked.connect(self.openCam)
        self.ui.Camera_Gather.clicked.connect(self.startGrab)
        self.ui.test.clicked.connect(self.test)
        self.ui.restart.clicked.connect(self.restart)
        self.ui.Camera_Close.clicked.connect(self.closeCam)
        self.ui.Model_Load.clicked.connect(self.model_Load)
        self.ui.Controller_Connect.clicked.connect(self.connected)
        self.ui.Controller_Experiment_15kg.clicked.connect(self.Verification_experiment_15kg)
        self.ui.Controller_Experiment_30kg.clicked.connect(self.Verification_experiment_30kg)
     
    #打开相机
    def openCam(self):
        self.thread.Task_choose('openCam')
        # self.thread.openCam()
        self.thread.signal.connect(self.callback)
        self.thread.start() 
        
    #图像采集
    def startGrab(self):  # 开始采集执行本函数
        self.thread.startGrab(self.ui.winid)
        self.thread.signal.connect(self.callback)
        self.thread.start() 
        
    # 关闭相机执行本函数
    def closeCam(self):  
        with open('./label.txt', encoding='utf-8' , mode='w') as f:      
            for i in self.data:
                f.write(i+'\n')
        # self.thread.Task_choose('closeCam')
        # self.thread.signal.connect(self.callback)
        # self.thread.start()  
        
    #连接控制器    
    def connected(self):
        self.thread.Task_choose('connected')
        self.thread.signal.connect(self.callback)
        self.thread.start()    # 启动线程
    
    #加载模型 
    def model_Load(self):
        self.thread.Task_choose('model_Load')
        self.thread.signal.connect(self.callback)
        self.thread.start()    # 启动线程
     
    #测试    
    def test(self):  
        self.thread.Task_choose('test')
        self.thread.signal.connect(self.callback)
        self.thread.image_signal.connect(self.callback_image)
        self.thread.start()    # 启动线程
    
    #系统复位 
    def restart(self):
        self.thread.Task_choose('restart')
        self.thread.signal.connect(self.callback)
        self.thread.start()    # 启动线程
        
    #15kg检定
    def Verification_experiment_15kg(self):
        self.thread.Task_choose('Verification_experiment_15kg')
        self.thread.signal.connect(self.callback)
        self.thread.image_signal.connect(self.callback_image)
        self.thread.start()    # 启动线程
    
     #30kg检定
    def Verification_experiment_30kg(self):
        self.thread.Task_choose('Verification_experiment_30kg')
        self.thread.signal.connect(self.callback)
        self.thread.image_signal.connect(self.callback_image)
        self.thread.start()    # 启动线程
        
    
    def callback(self, str):
        if str=='0':
            self.ui.event_message.append('控制器连接失败')
            self.ui.event_message.append("……………………") 
            msgBox = QMessageBox(QMessageBox.Warning, '提示', '控制器连接失败！')
            msgBox.exec()
            return
        elif str=='det':
            self.ui.event_message.append('检测模型加载失败')
            self.ui.event_message.append("……………………") 
            msgBox = QMessageBox(QMessageBox.Warning, '提示', '检测模型加载失败，请重新导入！')
            msgBox.exec()
            return 
        elif str=='rec':
            self.ui.event_message.append('识别模型加载失败')
            self.ui.event_message.append("……………………") 
            msgBox = QMessageBox(QMessageBox.Warning, '提示', '识别模型加载失败，请重新导入！')
            msgBox.exec()
            return 
        elif (str =="函数库初始化失败"):
            msgBox = QMessageBox(QMessageBox.Warning, '提示', '函数库初始化失败！')
            msgBox.exec()
            return
        elif (str=="查找连接计算机失败"):
            msgBox = QMessageBox(QMessageBox.Warning, '提示', '查找连接计算机失败！')
            msgBox.exec()
            return
        elif(str=="nCams.status"):
            msgBox = QMessageBox(QMessageBox.Warning, '提示', 'nCams.status')
            msgBox.exec()
            return
        elif(str== '没有找到相机,请确认连接和相机IP设置'):
            msgBox = QMessageBox(QMessageBox.Warning, '提示', '没有找到相机,请确认连接和相机IP设置!')
            msgBox.exec()
            return
        elif(str == '无法打开相机，可能正被别的软件控制!'):
            msgBox = QMessageBox(QMessageBox.Warning,'提示', '无法打开相机，可能正被别的软件控制!')
            msgBox.exec()
            return
        elif(str=='无法打开相机!'):
            msgBox = QMessageBox(QMessageBox.Warning, '提示', '无法打开相机!')
            msgBox.exec()
            return
        elif(str=='相机关闭出错'):
            msgBox = QMessageBox(QMessageBox.Warning, '提示', '相机关闭出错!')
            msgBox.exec()
            return
        elif str not in self.temp_str:
            self.ui.event_message.append(str)
            self.ui.event_message.append("……………………")   
            self.temp_str.append(str)
        else:
            return
  
    
    def callback_image(self, str):
        image = QImage(str)
        self.ui.result_image.setPixmap(QPixmap.fromImage(image))  # 加载图片
        self.ui.result_image.setScaledContents(True) # 设置图像自适应界面大小
           

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = UI_Logic_Window()
    myWin.show()
    sys.exit(app.exec_())    