import sys
import os
import torch
import time
from Camera.MVGigE import *
# import cv2
import shutil
from Verification_system.function import ZMCWrapper
# from model.config import config_model
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QMessageBox, QLabel, QFileDialog, QScrollArea, QComboBox, QLineEdit, QSlider, QGridLayout, QGroupBox, QCheckBox
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from Ui_ESACS import Ui_MainWindow #导入你写的界面类
from Camera.SingleGrab import MVCam
from Image_Processing.yolox_V1 import YOLO
from Image_Processing.model.DRNet_V8 import DRNet
from Image_Processing.digital_rec import digital_rec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UI_Logic_Window(QtWidgets.QMainWindow): 
    def __init__(self, parent =None):
        super(UI_Logic_Window,self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # self.winid = self.Image_Show.winId()  # 获取label对象的句柄
        self.ui.event_message.setText("系统初始化")
        self.ui.event_message.append("……………………………")
        self.save_dir = './result_image/'
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.out_floder='./Image_Processing/img_crop/'
        if not os.path.exists(self.out_floder):
            os.mkdir(self.out_floder)
        self.detect_dir = './Image_Processing/img/'
        if not os.path.exists(self.detect_dir):
            os.mkdir(self.detect_dir)
        
        self.num=11
       
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
     
    #连接控制器    
    def connected(self):
        ip="192.168.0.11"
        self.zaux = ZMCWrapper()
        ret=self.zaux.connect(ip)
        if(ret==0):
            self.ui.event_message.append('控制器连接成功!')
            self.ui.event_message.append("…………………………")
        else:
            msgBox = QMessageBox(QMessageBox.Warning, '提示', '控制器连接失败！')
            msgBox.exec()
            return
        
    #打开相机
    def openCam(self):
        r = MVInitLib()  # 初始化函数库
        if (r != MVSTATUS_CODES.MVST_SUCCESS):
            msgBox = QMessageBox(QMessageBox.Warning, '提示', '函数库初始化失败！')
            msgBox.exec()
            return
        r = MVUpdateCameraList()  # 查找连接到计算机上的相机
        if (r != MVSTATUS_CODES.MVST_SUCCESS):
            msgBox = QMessageBox(QMessageBox.Warning, '提示', '查找连接计算机失败！')
            msgBox.exec()
            return
        nCams = MVGetNumOfCameras()  # 获取相机数量
        if(nCams.status != MVSTATUS_CODES.MVST_SUCCESS):
            msgBox = QMessageBox(QMessageBox.Warning, '提示', nCams.status)
            msgBox.exec()
            return
        if(nCams.num == 0):
            msgBox = QMessageBox(QMessageBox.Warning, '提示', '没有找到相机,请确认连接和相机IP设置!')
            msgBox.exec()
            return
        hCam = MVOpenCamByIndex(0)  # 根据相机的索引返回相机句柄
        if(hCam.hCam == 0):
            if(hCam.status == MVSTATUS_CODES.MVST_ACCESS_DENIED):
                msgBox = QMessageBox(QMessageBox.Warning,'提示', '无法打开相机，可能正被别的软件控制!')
                msgBox.exec()
                return
            else:
                msgBox = QMessageBox(QMessageBox.Warning, '提示', '无法打开相机!')
                msgBox.exec()
                return
        w = MVGetWidth(hCam.hCam)  # 获取图像宽度
        h = MVGetHeight(hCam.hCam)  # 获取图像高度
        pf = MVGetPixelFormat(hCam.hCam)  # 获取图像格式
        self.hCam = hCam.hCam
        self.width = w.width
        self.height = h.height
        self.pixelFormat = pf.pixelFormat
        if(self.pixelFormat == MV_PixelFormatEnums.PixelFormat_Mono8):
            self.himage = MVImageCreate(self.width, self.height, 8).himage  # 创建图像句柄
        else:
            self.himage = MVImageCreate(self.width, self.height, 24).himage  # 创建图像句柄
        self.ui.event_message.append("相机已连接")
        self.ui.event_message.append("……………………………")
        
    #图像采集
    def startGrab(self):  # 开始采集执行本函数
        mode = MVGetTriggerMode(self.hCam)  # 获取当前相机采集模式
        source = MVGetTriggerSource(self.hCam)  # 获取当前相机信号源
        # if(self.sender().text() == '图像采集'):
        if(mode.pMode == TriggerModeEnums.TriggerMode_Off):  # 当触发模式关闭的时候，界面的行为
            # self.ui.Camera_Gather.setText('停止采集')
            MVStartGrabWindow(self.hCam, self.ui.winid)  # 将采集的图像传输到指定窗口
            self.ui.Camera_Connect.setEnabled(False)
            #self.combo.setEnabled(True)
            self.ui.Camera_Gather.setEnabled(True)
            #self.btnPause.setEnabled(True)
            #self.ui.Camera_Save.setEnabled(True)
            self.ui.Camera_Close.setEnabled(True)
        # else:
        #         if( source.source == TriggerSourceEnums.TriggerSource_Software):  # 当触发模式打开且为软触发的时候，界面的行为
        #             MVStartGrabWindow(self.hCam, self.ui.winid)  # 将采集的图像传输到指定窗口
        #             MVTriggerSoftware(self.hCam)
        #             # self.btnOpen.setEnabled(False)
        #             # self.combo.setEnabled(True)
        #             # self.btnStart.setEnabled(True)
        #             # self.btnPause.setEnabled(False)
        #             # self.btnSave.setEnabled(True)
        #             # self.btnSetting.setEnabled(True)
        #             # self.btnClose.setEnabled(True)
      

    def pauseGrab(self):  # 暂停或者继续执行本函数
        if(self.sender().text() == '继续采集'):
            # self.btnPause.setText('暂停采集')
            MVFreezeGrabWindow(self.hCam, False)  # 恢复图像传输到指定窗口
            # self.btnOpen.setEnabled(False)
            # self.combo.setEnabled(True)
            # self.btnStart.setEnabled(True)
            # self.btnPause.setEnabled(True)
            # self.btnSave.setEnabled(False)
            # self.btnSetting.setEnabled(False)
            # self.btnClose.setEnabled(True)
        else:
            # self.btnPause.setText('继续采集')
            MVFreezeGrabWindow(self.hCam, True)  # 暂停将图像传输到指定窗口
            # self.btnOpen.setEnabled(False)
            # self.combo.setEnabled(True)
            # self.btnStart.setEnabled(True)
            # self.btnPause.setEnabled(True)
            # self.btnSave.setEnabled(True)
            # self.btnSetting.setEnabled(False)
            # self.btnClose.setEnabled(True)
                    
    # 关闭相机执行本函数
    def closeCam(self):  
        result = MVCloseCam(self.hCam)
        if (result.status != MVSTATUS_CODES.MVST_SUCCESS):
            msgBox = QMessageBox(QMessageBox.Warning, '提示', result.status)
            msgBox.exec()
        self.ui.Camera_Connect.setEnabled(True)
        self.ui.Camera_Gather.setEnabled(False)
        self.ui.Camera_Close.setEnabled(False)

                
    def model_Load(self):
        #加载f分割模型
        det_weights='./Image_Processing/weights/YX_S-tiny.pth'
        if not os.path.exists(det_weights,):
            msgBox = QMessageBox(QMessageBox.Warning, '提示', '分割模型权重文件不存在或无法执行，请重新导入！')
            msgBox.exec()
            return
        else:
            self.det_model=YOLO(det_weights)
            # self.ui.event_message.append('分割模型加载成功')
            # self.ui.event_message.append("……………………………")
        #加载识别模型
        rec_weights='./Image_Processing/weights/drnet_v8_01.pt'
        if not os.access(rec_weights, os.X_OK):
            msgBox = QMessageBox(QMessageBox.Warning, '提示', '识别模型权重文件不存在或无法执行，请重新导入！')
            msgBox.exec()
            return   
        else:
            self.rec_model=DRNet(32, 1, 11, 256).to(device)
            self.rec_model.eval()
            self.rec_model.load_state_dict(torch.load(rec_weights))
            
        self.ui.event_message.append('模型加载成功')
        self.ui.event_message.append("……………………………")
            
        #处理
        # self.ui.event_message.append('图像处理开始')
        # self.ui.event_message.append("……………………………")
        # files=os.listdir(self.save_dir)
        # files.sort(key=lambda x: x.strip('.jpg').split('_')[-1])
        # with open('./result.txt', 'w') as f:
        #     for img in files:
        #         if img.split('.')[-1] in ['bmp', 'png', 'jpg', 'jpeg']:
        #             #开始分割
        #             image_path=os.path.join(self.save_dir, img)
        #             self.det_image, rec_result=self.det_model.detect_image(image_path, self.rec_model, crop=True)
        #             #显示检测结果
        #             self.det_image.save(os.path.join(self.detect_dir, img))
        #             self.origin_image= cv2.cvtColor(cv2.imread(os.path.join(self.detect_dir, img)), cv2.COLOR_BGR2BGRA)
        #             self.origin_QtImg = QtGui.QImage(self.origin_image.data, self.origin_image.shape[1], self.origin_image.shape[0], QtGui.QImage.Format_RGB32)
        #             self.ui.Image_Show.setPixmap(QtGui.QPixmap.fromImage(self.origin_QtImg))
        #             self.ui.Image_Show.setScaledContents(True) # 设置图像自适应界面大小
        #             #显示分割结果
        #             # self.det_model.detect_image(Image.open(image_path), crop=True)
        #             self.detect_image= cv2.cvtColor(cv2.imread(os.path.join(self.out_floder, img)), cv2.COLOR_BGR2BGRA)
        #             self.detect_QtImg = QtGui.QImage(self.detect_image.data, self.detect_image.shape[1], self.detect_image.shape[0], QtGui.QImage.Format_RGB32)
        #             self.ui.detect_image.setPixmap(QtGui.QPixmap.fromImage(self.detect_QtImg))
        #             self.ui.detect_image.setScaledContents(True) # 设置图像自适应界面大小
        #             self.ui.event_message.append(img+'分割完成')
        #             self.ui.event_message.append("……………………………")
        #             #开始识别
        #             # rec_result= digital_rec(self.out_floder+img,  self.rec_model)
        #             f.write(img+' '+ rec_result+"\n")
        #             self.ui.results_message.append(img+'：'+rec_result)
        #             self.ui.event_message.append(img+'识别完成')
        #             self.ui.event_message.append("……………………………") 
        #         else:
        #             msgBox = QMessageBox(QMessageBox.Warning, '提示', '图像读取出现问题！')
        #             msgBox.exec()
        #             return
    def SaveImage(self, image_name):
        idn = MVGetSampleGrab(self.hCam, self.himage)
        print(idn.idn)
        image_name=image_name+'.jpg'
        pathname = os.path.join(os.getcwd(), image_name)
        MVImageSave(self.himage, image_name.encode('utf-8'))  # 将图片保存下来
        #det_image, self.rec_result=self.det_model.detect_image(pathname, self.rec_model, crop=True)
       # if det_image==0:
            # return 
        self.num+=1
        shutil.move(pathname, os.path.join(self.save_dir, image_name))
        # if not os.path.exists(os.path.join(self.save_dir, image_name)):
        #     det_image.save(self.save_dir+ image_name)
        #     os.remove(pathname)
        # else:
        #     os.remove(self.save_dir+ image_name)
        #     det_image.save(self.save_dir+ image_name)
        #     os.remove(pathname)
        # if not os.path.exists(os.path.join(self.save_dir, image_name)):
        #     shutil.move(pathname, self.save_dir)
        # else:
        #     os.remove(self.save_dir+ image_name)
        #     shutil.move(pathname, self.save_dir)
        image = QImage(self.out_floder+ image_name)
        self.ui.result_image.setPixmap(QPixmap.fromImage(image))  # 加载图片
        self.ui.result_image.setScaledContents(True) # 设置图像自适应界面大小
    
    
    # 测试XYZ轴位移
    def test(self):  
        #设置轴属性
        self.SaveImage(str(self.num))
        # for i in range(5):
        #     self.zaux.set_atype(i, 1) #设置轴类型为脉冲轴
        #     if i>2:
        #         self.zaux.set_units(i, 800) #设置Z轴脉冲当量，一般设置成机台运动1mm需要的脉冲数
        #     else:
        #         self.zaux.set_units(i, 160) #设置Z轴脉冲当量，一般设置成机台运动1mm需要的脉冲数
        #     self.zaux.set_accel(i, 10) #设置轴运动速度
        #     self.zaux.set_decel(i, 2) #设置轴加速度
        #     self.zaux.set_speed(i, 2) #设置减速度速度
        # Z1轴运动位移
        # self.SaveImage(str(self.num).zfill(6))
        # self.num+=1
        Z1_Axis_List=[0, 1]
        Z1_Data_list=[100, 100]
        # Z1_Data_list_2=[147+120+18, 147+120+18]
        # Data_X=[35, -35, -35, 35, 0]
        # Data_Y=[-45, -45, 10, 10, 0]
        #self.zaux.multiAxis_moveAbs(2, Z1_Axis_List, Z1_Data_list) #加载平台上升
        # time.sleep(106)
         # XY轴运动
        # for i in range(4):
        #     self.zaux.singleAxis_moveAbs(4, Data_Y[i])  #Y轴运动位移
        #     time.sleep(20)
        #     self.zaux.singleAxis_moveAbs(3, Data_X[i]) #X轴运动位移
        #     time.sleep(30)
        # self.zaux.singleAxis_moveAbs(4, -10)  #Y轴运动位移
        # time.sleep(30)
        # self.restart()
        # #Z2轴运动位移
        # self.zaux.singleAxis_moveAbs(2, 50)
        # self.zaux.singleAxis_moveAbs(2, 0)
        
        #电动推杆测试
        # self.zaux.setCom_defaultBaud(9600, 8, 1, 0)
        # array_Instruction=[ "\"F0AN0\"" ,"\"F0AN1\"","\"F0AN2\"",
        #                                 "\"F0AN3\"","\"F0AN4\"","\"F0AN5\"",
        #                                 "\"F0AN6\"","\"F0AN7\"","\"F0AN8\"",
        #                                 "\"F0AN9\"","\"F0ANa\"","\"F0ANb\"",
        #                                 "\"F0ANc\"","\"F0ANd\"","\"F0ANe\""]
        # self.zaux.send_Data(0, "\"F0100\"")#砝码托盘下降
        # time.sleep(8)
        # self.SaveImage('Zero-'+str(0))
        # time.sleep(2)
        # for i in range(15): #闪变砝码下降
        #         self.zaux.send_Data(0, array_Instruction[i])
        #         time.sleep(13)
        #         self.SaveImage('Zero-'+str(i+1))
        #         print(self.rec_result)
        # self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部抬升
        # time.sleep(23)
        # self.zaux.send_Data(0, "\"F0101\"")#砝码托盘上升
        # time.sleep(8)
        # self.zaux.multiAxis_moveAbs(2, Z1_Axis_List, Z1_Data_list2) #加载平台下降
        # self.zaux.send_Data(0, "\"F0302\"")#砝码托盘下降
        # time.sleep(60)
        # self.zaux.send_Data(0, "\"F0303\"")#砝码托盘下降
        # time.sleep(8)
        
        # self.zaux.singleAxis_moveAbs(2, 200)
        #主砝码加载到5kg(一块主砝码子块)
        # self.zaux.multiAxis_moveAbs(2, Z1_Axis_List, Z1_Data_list_1) #加载平台上升
        # time.sleep(106)
        
        # Z_Axis_List=[0, 1, 2]
        # Z_Data_list=[147+120+20, 147+120+20, 120+20]
        # self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list) #加载平台上升, 辅助砝码同步上升
        #pass
        
    def restart(self):
        #设置轴属性
        for i in range(5):
            self.zaux.set_atype(i, 1) #设置轴类型为脉冲轴
            if i>2:
                self.zaux.set_units(i, 800) #设置Z轴脉冲当量，一般设置成机台运动1mm需要的脉冲数
            else:
                self.zaux.set_units(i, 160) #设置Z轴脉冲当量，一般设置成机台运动1mm需要的脉冲数
            self.zaux.set_accel(i, 10) #设置轴运动速度
            self.zaux.set_decel(i, 2) #设置轴加速度
            self.zaux.set_speed(i, 2) #设置减速度速度
        # Z1轴运动位移
        # Z1_Axis_List=[0, 1]
        # Z1_Data_list=[0, 0]
        # self.zaux.singleAxis_moveAbs(4, 0)
        # time.sleep(10)
        # self.zaux.singleAxis_moveAbs(3, 0)
        # time.sleep(10)
        # self.zaux.multiAxis_moveAbs(2, Z1_Axis_List, Z1_Data_list) #加载平台复位
        # self.zaux.singleAxis_moveAbs(2, 0)
        Z_Axis_List=[0, 1, 2, 3, 4]
        Z_Data_list=[0, 0, 0, 0, 0]
        self.zaux.multiAxis_moveAbs(5, Z_Axis_List, Z_Data_list) #加载平台上升, 辅助砝码同步上升
        self.zaux.setCom_defaultBaud(9600, 8, 1, 0)
        #self.zaux.send_Data(0, "\"F0000\"")#砝码托盘下降
        
        
        
    #15kg性能检定       
    def Verification_experiment_15kg(self):
        self.result_num=[]
        with open('./result.txt', encoding='utf-8' , mode='w') as f:      
            num=0   
            #设置轴属性
            for i in range(5):
                self.zaux.set_atype(i, 1) #设置轴类型为脉冲轴
                if i>2:
                    self.zaux.set_units(i, 800) #设置Z轴脉冲当量，一般设置成机台运动1mm需要的脉冲数
                else:
                    self.zaux.set_units(i, 160) #设置Z轴脉冲当量，一般设置成机台运动1mm需要的脉冲数
                self.zaux.set_accel(i, 10) #设置轴运动速度
                self.zaux.set_decel(i, 2) #设置轴加速度
                self.zaux.set_speed(i, 2) #设置减速度速度
                
            # 轴运动参数设置
            #轴号列表
            Z1_Axis_List=[0, 1] 
            Z_Axis_List=[0, 1, 2]  
            YX_Axis_list=[4, 3] #YX轴号列表
            ZYX_Axis_list=[0, 1, 2, 4, 3] #YX轴号列表
            #运动位置
            Z1_Data_list=[150, 150] #Z1轴上升距离
            Z_Data_list_up_5=[Z1_Data_list[0]+120+20, Z1_Data_list[1]+120+15, 120+15] #5kg加载距离
            Z_Data_list_down=[147, 147, 0]  #下降
            Z_Data_list_reset=[0, 0, 0]  #Z轴复位
            ZYX_Data_list_reset=[0, 0, 0, 0, 0]  #XYZ轴复位
            Data_X=[35, -35, -35, 35, 0] #四点X轴坐标
            Data_Y=[-45, -45, 10, 10, 0]  #四点Y轴坐标'
            #闪变砝码加载指令列表
            array_Instruction=[ "\"F0AN0\"" ,"\"F0AN1\"","\"F0AN2\"",
                                            "\"F0AN3\"","\"F0AN4\"","\"F0AN5\"",
                                            "\"F0AN6\"","\"F0AN7\"","\"F0AN8\"",
                                            "\"F0AN9\"","\"F0ANa\"","\"F0ANb\"",
                                            "\"F0ANc\"","\"F0ANd\"","\"F0ANe\""]
            self.zaux.setCom_defaultBaud(9600, 8, 1, 0) #开启串口通信
                        
            f.write('15kg量程电子秤检定实验'+"\n")
            
            #15kg量程-置零性检定
            f.write('************置零检定开始************'+"\n") #记录砝码托盘下降示值
            self.ui.event_message.append('置零检定开始')
            self.ui.event_message.append("……………………")   
            self.zaux.multiAxis_moveAbs(2, Z1_Axis_List, Z1_Data_list) #ZI1轴平台上升
            time.sleep(106)
            self.SaveImage('Zero-'+str(num))
            time.sleep(2)
            num+=1
            f.write('Zero-'+str(num)+'='+self.rec_result +"\n")
            self.zaux.send_Data(0, "\"F0100\"")#砝码托盘下降
            time.sleep(8) 
            self.SaveImage('Zero-'+str(num))
            time.sleep(2)
            num+=1
            f.write('Zero-'+str(num)+'='+self.rec_result +"\n") #记录砝码托盘下降示值
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                self.SaveImage('Zero-'+str(num)) #采集图像
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Zero-s-'+str(i+1)+'='+self.rec_result +"\n")
                    f.write('Zero Change='+str(i+1)+'\n')
                    break
                else:
                    if((i+1)!=15):   
                        f.write('Zero-s-'+str(i+1)+'='+self.rec_result +"\n")
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('Zero Checking NO'+'\n')         
            self.zaux.send_Data(0, "\"F00U0\"")#砝码托盘下降
            time.sleep(23)     
            self.result_num.clear() 
            self.ui.event_message.append('置零检定结束')
            self.ui.event_message.append("……………………………")
            f.write('************置零检定结束************'+"\n") #记录砝码托盘下降示值
            num=0
            self.ui.event_message.append('置零检定结束')
            self.ui.event_message.append("……………………")   
    
            #15kg量程-偏载性检定
            f.write('\n************偏载检定开始************'+"\n") #
            self.ui.event_message.append('偏载检定开始')
            self.ui.event_message.append("……………………")   
            
            #偏载性能左上角检定
            f.write('************Left_Top Start************'+"\n") 
            self.zaux.send_Data(0,"\"F0300\"")#前置偏载砝码下降
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0101\"")#砝码托盘上升
            time.sleep(8)
            #XY轴运动到左上角，Y轴先动
            Left_Top_list=[Data_Y[0], Data_X[0]] #运动位移
            for i in range(len(YX_Axis_list)):
                self.zaux.singleAxis_moveAbs(YX_Axis_list[i], Left_Top_list[i])
                time.sleep(30)            
            self.zaux.send_Data(0,"\"F0100\"")#砝码托盘下降
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
            time.sleep(23)
            #主砝码加载到5kg(一块主砝码子块)
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_5) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('Unbalance_Left_Top-'+str(num))
            time.sleep(2)
            f.write('Unbalance_Left_Top-'+str(num)+'='+self.rec_result +"\n")
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('Unbalance_Left_Top-s-'+str(i+1))
                time.sleep(2)
                f.write('Unbalance_Left_Top-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('Unbalance_Left_Top Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0201\"")#置零砝码上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0101\"")#砝码托盘上升
            time.sleep(23)
            #主砝码卸载(一块主砝码子块)
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_down) #加载平台下降, 辅助砝码同步下降
            time.sleep(120)
            num=0
            self.ui.event_message.append('左上角检定结束')
            f.write('************Left Top End ************'+"\n") 
            
            ##偏载性能右上角检定
            f.write('************Right Top Start************'+"\n") 
            #XY轴运动到右上角，Y轴先动
            Right_Top_list=[Data_Y[1], Data_X[1]] #运动位移
            for i in range(len(YX_Axis_list)):
                self.zaux.singleAxis_moveAbs(YX_Axis_list[i], Right_Top_list[i])
                time.sleep(30)            
            self.zaux.send_Data(0,"\"F0100\"")#砝码托盘下降
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
            time.sleep(23)
            #主砝码加载到5kg(一块主砝码子块)
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_5) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('Unbalance_Right_Top-'+str(num))
            time.sleep(2)
            f.write('Unbalance_Right_Top-'+str(num)+'='+self.rec_result +"\n")
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('Unbalance_Right_Top-s-'+str(i+1))
                time.sleep(2)
                f.write('Unbalance_Right_Top-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('Unbalance_Right_Top Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0201\"")#置零砝码上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0101\"")#砝码托盘上升
            time.sleep(23)
            #主砝码卸载(一块主砝码子块
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_down) #加载平台下降, 辅助砝码同步下降
            time.sleep(120)
            num=0
            self.ui.event_message.append('右上角检定结束')
            f.write('************Right Top End************'+"\n") 
          
            #XY轴运动到原点，Y轴先动
            Center_list=[Data_Y[4], Data_X[4]] #运动位移
            for i in range(len(YX_Axis_list)):
                self.zaux.singleAxis_moveAbs(YX_Axis_list[i], Center_list[i])
                time.sleep(30)      
            self.zaux.send_Data(0,"\"F0302\"")#后置偏载砝码下降
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0301\"")#前置偏载砝码上升
            time.sleep(23)
            
            #偏载性能右下角检定
            f.write('************Right Bottom Start************'+"\n") 
            #XY轴运动到右下角，Y轴先动
            Right_Bottom_list=[Data_Y[2], Data_X[2]] #运动位移
            for i in range(len(YX_Axis_list)):
                self.zaux.singleAxis_moveAbs(YX_Axis_list[i], Right_Bottom_list[i])
                time.sleep(30)            
            self.zaux.send_Data(0,"\"F0100\"")#砝码托盘下降
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
            time.sleep(23)
            #主砝码加载到5kg(一块主砝码子块)
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_5) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('Unbalance_Right_Bottom-'+str(num))
            time.sleep(2)
            f.write('Unbalance_Right_Bottom-'+str(num)+'='+self.rec_result +"\n")
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('Unbalance_Right_Bottom-s-'+str(i+1))
                time.sleep(2)
                f.write('Unbalance_Right_Bottom-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('Unbalance_Right_Bottom Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0201\"")#置零砝码上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0101\"")#砝码托盘上升
            time.sleep(23)
            #主砝码卸载(一块主砝码子块
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_down) #加载平台下降, 辅助砝码同步下降
            time.sleep(120)
            num=0
            self.ui.event_message.append('右下角检定结束')
            f.write('************Right Bottom End************'+"\n") 
            
            # 偏载性能左下角检定
            f.write('************Left Bottom Start************'+"\n") 
            #XY轴运动到右下角，Y轴先动
            Left_Bottom_list=[Data_Y[3], Data_X[3]] #运动位移
            for i in range(len(YX_Axis_list)):
                self.zaux.singleAxis_moveAbs(YX_Axis_list[i], Left_Bottom_list[i])
                time.sleep(30)            
            self.zaux.send_Data(0,"\"F0100\"")#砝码托盘下降
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
            time.sleep(23)
            #主砝码加载到5kg(一块主砝码子块)
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_5) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('Unbalance_Left_Bottom-'+str(num))
            time.sleep(2)
            f.write('Unbalance_Left_Bottom-'+str(num)+'='+self.rec_result +"\n")
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('Unbalance_Left_Bottom-s-'+str(i+1))
                time.sleep(2)
                f.write('Unbalance_Left_Bottom-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('Unbalance_Left_Bottom Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0201\"")#置零砝码上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0101\"")#砝码托盘上升
            time.sleep(23)
            #主砝码卸载(一块主砝码子块
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_down) #加载平台下降, 辅助砝码同步下降
            time.sleep(120)
            num=0
            self.ui.event_message.append('左下角检定结束')
            f.write('************Left Bottom End************'+"\n") 
                        
            #偏载性能中心位置检定
            f.write('************Center Start************'+"\n") 
            #XY轴运动到中心位置，Y轴先动
            Center_list=[Data_Y[4], Data_X[4]] #运动位移
            for i in range(len(YX_Axis_list)):
                self.zaux.singleAxis_moveAbs(YX_Axis_list[i], Center_list[i])
                time.sleep(30)            
            self.zaux.send_Data(0,"\"F0100\"")#砝码托盘下降
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
            time.sleep(23)
            #主砝码加载到5kg(一块主砝码子块)
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_5) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('Unbalance_Center-'+str(num))
            time.sleep(2)
            f.write('Unbalance_Center-'+str(num)+'='+self.rec_result +"\n")
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('Unbalance_Center-s-'+str(i+1))
                time.sleep(2)
                f.write('Unbalance_Center-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('Unbalance_Center Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0201\"")#置零砝码上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0101\"")#砝码托盘上升
            time.sleep(23)
            #主砝码卸载(一块主砝码子块
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_down) #加载平台下降, 辅助砝码同步下降
            time.sleep(120)
            num=0
            self.ui.event_message.append('中心位置检定结束')
            f.write('************Center End************'+"\n") 
           
            
            # #置零准确度命令
            # self.zaux.send_Data(0,"\"F0100\"")#砝码托盘下降
            # time.sleep(23)
            
            # #称量性能检定
            # for i in range(15): #闪变砝码下降
            #     self.zaux.send_Data(0, array_Instruction[i])
            #     time.sleep(14)
            # self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
            # time.sleep(23)
            # self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
            # time.sleep(23)
            # self.zaux.send_Data(0,"\"F0300\"")#前置偏载砝码下降
            # time.sleep(23)
            # self.zaux.send_Data(0,"\"F0201\"")#置零砝码上升
            # time.sleep(23)
            # self.zaux.send_Data(0,"\"F0301\"")#前置偏载砝码上升
            # time.sleep(23)
            
            # #除皮误差性能检定
            # self.zaux.send_Data(0,"\"F0402\"")#除皮2砝码下降
            # time.sleep(23)
            # self.zaux.send_Data(0,"\"F0600\"")#除皮功能按键
            # time.sleep(23)
            
            # #50g除皮置零检定
            # self.zaux.send_Data(0,"\"F0100\"")#砝码托盘下降
            # time.sleep(23)
            # for i in range(15): #闪变砝码下降
            #     self.zaux.send_Data(0, array_Instruction[i])
            #     time.sleep(14)
            # self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
            # time.sleep(23)
            
            # #2.5kg除皮误差性能检定
            # self.zaux.send_Data(0,"\"F0301\"")#前置偏载砝码下降
            # time.sleep(23)
            # self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
            # time.sleep(23)
            # for i in range(15): #闪变砝码下降
            #     self.zaux.send_Data(0, array_Instruction[i])
            #     time.sleep(14)
            # self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
            # time.sleep(23)
            
            # #5kg除皮误差性能检定
            # for i in range(15): #闪变砝码下降
            #     self.zaux.send_Data(0, array_Instruction[i])
            #     time.sleep(14)
            # self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
            # time.sleep(23)
            # #7.5kg除皮误差性能检定
            # for i in range(15): #闪变砝码下降
            #     self.zaux.send_Data(0, array_Instruction[i])
            #     time.sleep(14)
            # self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
            # time.sleep(23)
            # #10kg除皮误差性能检定
            # for i in range(15): #闪变砝码下降
            #     self.zaux.send_Data(0, array_Instruction[i])
            #     time.sleep(14)
            # self.zaux.send_Data(0,"\"F0000\"")#闪变砝码全部上升
            # time.sleep(23)
        
            # #置零按键功能(同称量性能检定，选择30kg或接近30kg称量点)
            # self.zaux.send_Data(0,"\"F0601\"")#闪变砝码全部上升
            # time.sleep(23)
            
            # #鉴别阈值的检测应该在三个不同载荷下进行检定，分别是 min、max/2、max 三种情况下，故选取5g、10kg 与 15kg三种质量载荷放置
            # self.zaux.send_Data(0,"\"F0100\"")#砝码托盘下落
            # time.sleep(23)
            # self.zaux.send_Data(0,"\"F0200\"")#置零砝码下落
            # time.sleep(23)
            # self.zaux.send_Data(0,"\"F0301\"")#前置零偏载砝码下降
            # time.sleep(23)
            # self.zaux.send_Data(0,"\"F0504\"")#00-09闪变砝码下落
            # time.sleep(23)
            
            # array_Instruction_2=[ "\"F0BN9\"" ,"\"F0BN8\"","\"F0BN7\"",
            #                         "\"F0BN6\"","\"F0BN5\"","\"F0BN4\"",
            #                         "\"F0BN3\"","\"F0BN2\"","\"F0BN1\"",
            #                         "\"F0BN0\""]
            # for i in range(len(array_Instruction_2)): #闪变砝码下降
            #     self.zaux.send_Data(0, array_Instruction_2[i])
            #     time.sleep(14)
            # self.zaux.send_Data(0,"\"F0AN7\"")#最后一个闪变砝码加载
            # time.sleep(23)
            # self.zaux.send_Data(0,"\"F0500\"")#鉴别阈砝码下落
            # time.sleep(23)
            # self.zaux.send_Data(0,"\"F0501\"")#鉴别阈砝码上升
            # time.sleep(23)
            # self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
            # time.sleep(23)
            
            # #所有检定结束，部件归零
            # self.zaux.send_Data(0,"\"F0000\"")#闪变砝码全部上升
            # time.sleep(23)
        
    #30kg性能检定       
    def Verification_experiment_30kg(self):
        #置零性能检定
        array_Instruction=[ "\"F0AN0\"" ,"\"F0AN1\"","\"F0AN2\"",
                                    "\"F0AN3\"","\"F0AN4\"","\"F0AN5\"",
                                    "\"F0AN6\"","\"F0AN7\"","\"F0AN8\"",
                                    "\"F0AN9\"","\"F0ANa\"","\"F0ANb\"",
                                    "\"F0ANc\"","\"F0ANd\"","\"F0ANe\""]
        #置零性能检定
        self.zaux.send_Data(0,"\"FZERO\"")#启动开始按钮
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0100\"")#砝码托盘下降
        time.sleep(23)
        for i in range(len(array_Instruction)): #闪变砝码下降
            self.zaux.send_Data(0, array_Instruction[i])
            time.sleep(14)
        self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部抬升
        time.sleep(23)
        
        #偏载性能检定
        #左上角
        self.zaux.send_Data(0,"\"F0300\"")#前置偏载砝码下降
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0101\"")#砝码托盘上升
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0100\"")#砝码托盘下降
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
        time.sleep(23)
        for i in range(len(array_Instruction)): #闪变砝码下降
            self.zaux.send_Data(0, array_Instruction[i])
            time.sleep(14)
        self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0201\"")#置零砝码上升
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0101\"")#砝码托盘上升
        time.sleep(23)
        #右上角
        self.zaux.send_Data(0,"\"F0100\"")#砝码托盘下降
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
        time.sleep(23)
        for i in range(len(array_Instruction)): #闪变砝码下降
            self.zaux.send_Data(0, array_Instruction[i])
            time.sleep(14)
        self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0201\"")#置零砝码上升
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0101\"")#砝码托盘上升
        time.sleep(23)
        
        #偏载砝码更换
        self.zaux.send_Data(0,"\"F0302\"")#后置偏载砝码下降
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0301\"")#前置偏载砝码上升
        time.sleep(23)
        
        #右下角
        self.zaux.send_Data(0,"\"F0100\"")#砝码托盘下降
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
        time.sleep(23)
        for i in range(len(array_Instruction)): #闪变砝码下降
            self.zaux.send_Data(0, array_Instruction[i])
            time.sleep(14)
        self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0201\"")#置零砝码上升
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0100\"")#砝码托盘上升
        time.sleep(23)
        
        #左下角
        self.zaux.send_Data(0,"\"F0100\"")#砝码托盘下降
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
        time.sleep(23)
        for i in range(len(array_Instruction)): #闪变砝码下降
            self.zaux.send_Data(0, array_Instruction[i])
            time.sleep(14)
        self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0201\"")#置零砝码上升
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0100\"")#砝码托盘上升
        time.sleep(23)
        
        #中心位置
        self.zaux.send_Data(0,"\"F0100\"")#砝码托盘下降
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
        time.sleep(23)
        for i in range(len(array_Instruction)): #闪变砝码下降
            self.zaux.send_Data(0, array_Instruction[i])
            time.sleep(14)
        self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0201\"")#置零砝码上升
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0305\"")#后置偏载砝码上升
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0100\"")#砝码托盘上升
        time.sleep(23)
        
        #开始做称量性能检定之前需要一次置零准确度命令
        self.zaux.send_Data(0,"\"FZERO\"")#后置偏载砝码上升
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0100\"")#砝码托盘上升
        time.sleep(23)
        
        #称量性能检定
        for i in range(len(array_Instruction)): #闪变砝码下降
            self.zaux.send_Data(0, array_Instruction[i])
            time.sleep(14)
        self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0300\"")#前置偏载砝码下降
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0201\"")#置零砝码上升
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0301\"")#前置偏载砝码下降
        time.sleep(23)
        
        #除皮性能检定
        self.zaux.send_Data(0,"\"F0402\"")#除皮4砝码下降
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0600\"")#除皮功能按键
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0401\"")#除皮4砝码上升
        time.sleep(23)
        #100g砝码除皮检定
        self.zaux.send_Data(0,"\"F0100\"")#除皮4砝码下降
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
        time.sleep(23)
        for i in range(len(array_Instruction)): #闪变砝码下降
            self.zaux.send_Data(0, array_Instruction[i])
            time.sleep(14)
        self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
        time.sleep(23)   
        #2.5kg除皮误差检定
        self.zaux.send_Data(0,"\"F0301\"")#前置偏载砝码下降
        time.sleep(23)
        for i in range(len(array_Instruction)): #闪变砝码下降
            self.zaux.send_Data(0, array_Instruction[i])
            time.sleep(14)
        self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
        time.sleep(23) 
        #5kg除皮砝码检定
        for i in range(len(array_Instruction)): #闪变砝码下降
            self.zaux.send_Data(0, array_Instruction[i])
            time.sleep(14)
        self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
        time.sleep(23)   
        #7.5kg除皮砝码检定
        for i in range(len(array_Instruction)): #闪变砝码下降
            self.zaux.send_Data(0, array_Instruction[i])
            time.sleep(14)
        self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
        time.sleep(23)   
        #10kg除皮砝码检定
        for i in range(len(array_Instruction)): #闪变砝码下降
            self.zaux.send_Data(0, array_Instruction[i])
            time.sleep(14)
        self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
        time.sleep(23)   
        #重复性能检定
        self.zaux.send_Data(0,"\"F0601\"")#闪变砝码全部上升
        time.sleep(23)
        #鉴别阈性能检定
        array_Instruction_2=[ "\"F0BN13\"" ,"\"F1AN12\"","\"F1AN10\"",
                                          "\"F1AN9\"","\"F1AN8\"","\"F1AN7\""]
        #5g除皮砝码检定
        for i in range(len(array_Instruction)): #闪变砝码逐个下降
            self.zaux.send_Data(0, array_Instruction[i])
            time.sleep(14)
        self.zaux.send_Data(0,"\"F0505\"")#闪变砝码加载
        time.sleep(23)
        for i in range(len(array_Instruction_2)): #砝码逐个上升
            self.zaux.send_Data(0, array_Instruction[i])
            time.sleep(14)
        self.zaux.send_Data(0,"\"F0502\"")#鉴别阈值砝码下降
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0503\"")#鉴别阈值砝码上升
        time.sleep(23)
        self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
        time.sleep(23)
        #10kg
        self.zaux.send_Data(0,"\"F0505\"")#闪变砝码加载
        time.sleep(23)
        for i in range(len(array_Instruction_2)): #砝码逐个上升
            self.zaux.send_Data(0, array_Instruction[i])
            time.sleep(14)
        self.zaux.send_Data(0,"\"F0502\"")#鉴别阈值砝码下落
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0503\"")#鉴别阈值砝码上升
        time.sleep(23)
        self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
        time.sleep(23)
        #15kg
        self.zaux.send_Data(0,"\"F0505\"")#闪变砝码加载
        time.sleep(23)
        for i in range(len(array_Instruction_2)): #砝码逐个上升
            self.zaux.send_Data(0, array_Instruction[i])
            time.sleep(14)
        self.zaux.send_Data(0,"\"F0502\"")#鉴别阈值砝码下落
        time.sleep(23)
        self.zaux.send_Data(0,"\"F0503\"")#鉴别阈值砝码上升
        time.sleep(23)
        self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
        time.sleep(23)
        
        #检定结束
        self.zaux.send_Data(0,"\"F0000\"")#所有部件归零位
        time.sleep(23)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = UI_Logic_Window()
    myWin.show()
    sys.exit(app.exec_())    