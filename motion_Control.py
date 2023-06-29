from Verification_system.function import ZMCWrapper
import torch
import time, os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import random
from Camera.MVGigE import *
from PyQt5.QtWidgets import *
from Image_Processing.digital_rec import digital_rec
from Image_Processing.yolox_V1 import YOLO
from Image_Processing.model.DRNet_V8 import DRNet
from Camera.SingleGrab import MVCam
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class  motion_Control(QThread):
    #声明一个信号
    signal=pyqtSignal(str)
    # camera_signal=pyqtSignal(int)
    image_signal=pyqtSignal(str)
    #初始化
    def __init__(self):
        super(motion_Control, self).__init__()
        self.save_dir = './result_image/'
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.out_floder='./Image_Processing/img_crop/'
        if not os.path.exists(self.out_floder):
            os.mkdir(self.out_floder)
        self.detect_dir = './Image_Processing/img/'
        if not os.path.exists(self.detect_dir):
            os.mkdir(self.detect_dir)
        self.message=' '
        self.image_name=''
     
    #任务选择 
    def Task_choose(self, status):
        self.status=status   
        
    #打开相机
    def openCam(self):
        r = MVInitLib()  # 初始化函数库
        if (r != MVSTATUS_CODES.MVST_SUCCESS):
            self.message="函数库初始化失败"
        r = MVUpdateCameraList()  # 查找连接到计算机上的相机
        if (r != MVSTATUS_CODES.MVST_SUCCESS):
            self.message="查找连接计算机失败"
        nCams = MVGetNumOfCameras()  # 获取相机数量
        if(nCams.status != MVSTATUS_CODES.MVST_SUCCESS):
            self.message="nCams.status"
        if(nCams.num == 0):
            self.message='没有找到相机,请确认连接和相机IP设置'
        hCam = MVOpenCamByIndex(0)  # 根据相机的索引返回相机句柄
        if(hCam.hCam == 0):
            if(hCam.status == MVSTATUS_CODES.MVST_ACCESS_DENIED):
                self.message='无法打开相机，可能正被别的软件控制!'
                pass
            else:
                # self.message='无法打开相机!'
                pass
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
        self.message='相机已连接'
        
    # 图像采集
    def startGrab(self, winid):  # 开始采集执行本函数
        self.message='图像已采集'
        mode = MVGetTriggerMode(self.hCam)  # 获取当前相机采集模式
        source = MVGetTriggerSource(self.hCam)  # 获取当前相机信号源
        MVStartGrabWindow(self.hCam, winid)  # 将采集的图像传输到指定窗口
        # if(self.sender().text() == '图像采集'):
        # if(mode.pMode == TriggerModeEnums.TriggerMode_Off):  # 当触发模式关闭的时候，界面的行为
            # self.ui.Camera_Gather.setText('停止采集')
            # MVStartGrabWindow(self.hCam, winid)  # 将采集的图像传输到指定窗口
            
                    
    #关闭相机执行本函数
    def closeCam(self):  
        result = MVCloseCam(self.hCam)
        if (result.status != MVSTATUS_CODES.MVST_SUCCESS):
            self.message='相机关闭出错'
            # msgBox = QMessageBox(QMessageBox.Warning, '提示', result.status)
            # msgBox.exec()
        self.message='相机已关闭'
    
    def SaveImage(self, image_name):
        idn = MVGetSampleGrab(self.hCam, self.himage)
        print(idn.idn)
        image_name=image_name+'.jpg'
        pathname = os.path.join(os.getcwd(), image_name)
        MVImageSave(self.himage, image_name.encode('utf-8'))  # 将图片保存下来
        det_image, self.rec_result=self.det_model.detect_image(pathname, self.rec_model, crop=True)
        if det_image==0:
            shutil.move(pathname, os.path.join(self.save_dir, image_name))
            return 
        if not os.path.exists(os.path.join(self.save_dir, image_name)):
            det_image.save(self.save_dir+ image_name)
            os.remove(pathname)
        else:
            os.remove(self.save_dir+ image_name)
            det_image.save(self.save_dir+ image_name)
            os.remove(pathname) 
        self.image_name=self.save_dir+ image_name
        self.image_signal.emit(self.image_name) 
        
    #加载模型 
    def model_Load(self):
        #加载f分割模型
        det_weights='./Image_Processing/weights/YX_S-tiny_02.pth'
        rec_weights='./Image_Processing/weights/drnet_v8_03.pt'
        if not os.path.exists(det_weights):
            self.message='det'
        else:
            self.det_model=YOLO(det_weights)
              #加载识别模型
        if not os.path.exists(rec_weights):
            self.message='rec'
        else:
            self.rec_model=DRNet(32, 1, 11, 256).to(device)
            self.rec_model.eval()
            self.rec_model.load_state_dict(torch.load(rec_weights))   
            self.message='模型加载成功'
      
                
    #连接控制器    
    def connected(self):
        #设置ip连接
        ip="192.168.0.11"
        self.zaux = ZMCWrapper()
        if(self.zaux.connect(ip)==0):
            self.message='控制器已连接'
        else:
            self.message='0'
            
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
      
    #测试函数   
    def test(self):
        # Z1轴运动位移
        Z1_Axis_List=[0, 1]
        # Z1_Data_list_1=[147, 147]
        # Z1_Data_list_2=[147+120+18, 147+120+18]
        Data_X=[35, -35, -35, 35, 0]
        Data_Y=[-45, -45, 10, 10, 0]
        # time.sleep(100)
        # self.message='测试动作完成'
        # self.SaveImage('test')
        Z_Axis_List=[0, 1, 2]  
        Z1_Data_list=[105, 105] #Z1轴上升距离
        Z_Data_list_up_2=[Z1_Data_list[0]+120+5, Z1_Data_list[1]+120+5, 120+5] #2.5kg加载距离
        Z_Data_list_up_5=[Z1_Data_list[0]+120+10, Z1_Data_list[1]+120+10, 120+10] #5kg加载距离
        Z_Data_list_up_7=[Z1_Data_list[0]+120+15, Z1_Data_list[1]+120+15, 120+15] #7.5kg加载距离
        Z_Data_list_up_10=[Z1_Data_list[0]+120+20, Z1_Data_list[1]+120+20, 120+20] #10kg加载距离
        Z_Data_list_up_15=[Z1_Data_list[0]+120+25, Z1_Data_list[1]+120+25, 120+25] #15kg加载距离
        Z_Data_list_down=[Z1_Data_list[0], Z1_Data_list[1], 0]
        self.signal.emit('测试1')
        self.zaux.multiAxis_moveAbs(2, Z_Axis_List , Z1_Data_list) #加载平台上升
        time.sleep(80)
        self.signal.emit('测试1完成')
        self.zaux.setCom_defaultBaud(9600, 8, 1, 0) #开启串口通信      
        self.zaux.send_Data(0, "\"F0100\"")#砝码托盘下降
        time.sleep(23)   
        # time.sleep(100)
        # self.signal.emit('测试2')
        # self.zaux.multiAxis_moveAbs(3, Z_Axis_List ,Z_Data_list_up_2) #加载平台上升
        # time.sleep(100)
        # self.signal.emit('测试3')
        # self.zaux.multiAxis_moveAbs(3, Z_Axis_List ,Z_Data_list_up_5) #加载平台上升
        # time.sleep(100)
        # self.signal.emit('测试4')
        # self.zaux.multiAxis_moveAbs(3, Z_Axis_List ,Z_Data_list_up_7) #加载平台上升
        # time.sleep(100)
        # self.signal.emit('测试5')
        # self.zaux.multiAxis_moveAbs(3, Z_Axis_List ,Z_Data_list_up_10) #加载平台上升
        # time.sleep(100)
        # self.signal.emit('测试6')
        # self.zaux.multiAxis_moveAbs(3, Z_Axis_List ,Z_Data_list_up_15) #加载平台上升
        # time.sleep(100)
        # self.signal.emit('测试7')
        # self.zaux.multiAxis_moveAbs(3, Z_Axis_List ,Z_Data_list_up_10) #加载平台上升
        # time.sleep(100)
        # self.signal.emit('测试8')
        # self.zaux.multiAxis_moveAbs(3, Z_Axis_List ,Z_Data_list_up_15) #加载平台上升
        # time.sleep(100)
    
    #复位
    def restart(self):            
        Z_Axis_List=[0, 1, 2, 3, 4]
        Z_Data_list=[0, 0, 0, 0, 0]
        self.zaux.multiAxis_moveAbs(5, Z_Axis_List, Z_Data_list) #加载平台上升, 辅助砝码同步上升
        self.zaux.setCom_defaultBaud(9600, 8, 1, 0)
        self.zaux.send_Data(0, "\"F0000\"")#砝码托盘下降
        time.sleep(80)
        self.message='系统各模块已复位'
      
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
            Z1_Data_list=[112, 112] #Z1轴上升距离
            Z_Data_list_up_2=[Z1_Data_list[0]+120+5, Z1_Data_list[1]+120+5, 120+5] #2.5kg加载距离
            Z_Data_list_up_5=[Z1_Data_list[0]+120+10, Z1_Data_list[1]+120+10, 120+10] #5kg加载距离
            Z_Data_list_up_7=[Z1_Data_list[0]+120+15, Z1_Data_list[1]+120+15, 120+15] #7.5kg加载距离
            Z_Data_list_up_10=[Z1_Data_list[0]+120+20, Z1_Data_list[1]+120+20, 120+20] #10kg加载距离
            Z_Data_list_up_15=[Z1_Data_list[0]+120+25, Z1_Data_list[1]+120+25, 120+25] #15kg加载距离
            Z_Data_list_down=[Z1_Data_list[0], Z1_Data_list[1], 0]
            
            
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
            
            array_Instruction_2=[ "\"F0BN9\"" ,"\"F0BN8\"","\"F0BN7\"",
                                            "\"F0BN6\"","\"F0BN5\"","\"F0BN4\"",
                                            "\"F0BN3\"","\"F0BN2\"","\"F0BN1\"",
                                            "\"F0BN0\""]
            
            self.zaux.setCom_defaultBaud(9600, 8, 1, 0) #开启串口通信      
            f.write('Verification_experiment_15kg'+"\n")
            self.signal.emit('15kg量程电子秤检定实验')
            #15kg量程-置零性检定
            f.write('************Zero Verification************'+"\n") #记录砝码托盘下降示值
            self.signal.emit('01-置零检定开始')
            self.zaux.multiAxis_moveAbs(2, Z1_Axis_List, Z1_Data_list) #ZI1轴平台上升
            time.sleep(80)
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
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码上升
            time.sleep(23)     
            self.result_num.clear() 
            self.signal.emit('01-置零检定已完成')
            num=0
    
            #15kg量程-偏载性检定
            f.write('\n************Off-load calibration************'+"\n") #
            self.signal.emit('02-偏载检定开始')
    
            #偏载性能左上角检定
            f.write('************Left_Top************'+"\n") 
            self.signal.emit('02-偏载检定-左上角')
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
            time.sleep(110)
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
            time.sleep(110)
            num=0
            
            
            ##偏载性能右上角检定
            f.write('************Right Top************'+"\n") 
            self.signal.emit('02-偏载检定-右上角')
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
            self.signal.emit('02-偏载检定-右下角')
            f.write('************Right Bottom************'+"\n") 
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
            
            
            # 偏载性能左下角检定
            self.signal.emit('02-偏载检定-左下角')
            f.write('************Left Bottom************'+"\n") 
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
            
                        
            #偏载性能中心位置检定
            self.signal.emit('02-偏载检定-中心位置')
            f.write('************Center************'+"\n") 
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
            self.zaux.send_Data(0,"\"F0303\"")#后置置砝码上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0101\"")#砝码托盘上升
            time.sleep(23)
            #主砝码卸载(一块主砝码子块
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_down) #加载平台下降, 辅助砝码同步下降
            time.sleep(120)
            num=0
            self.signal.emit('02-偏载检定完成')
            
            
            #置零准确度检定
            f.write('************Zero Verification************'+"\n") #记录砝码托盘下降示值
            self.signal.emit('03-置零检定开始')
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
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码抬升
            time.sleep(23)     
            self.result_num.clear() 
            self.signal.emit('03-置零检定已完成')
            num=0
            
            #称量性能检定
            f.write('************Weighing Performance Verification************'+"\n") #记录砝码托盘下降示值
            self.signal.emit('04-称量性能检定开始')
            self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
            time.sleep(23)
            self.SaveImage('W-'+str(num))
            f.write('W-'+str(num)+'='+self.rec_result +"\n") #记录置零砝码下落示值
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                self.SaveImage('W-'+str(num)) #采集图像
                if self.rec_result not in self.result_num and i!=0:
                    f.write('W-s-'+str(i+1)+'='+self.rec_result +"\n")
                    f.write('W Change='+str(i+1)+'\n')
                    break
                else:
                    if((i+1)!=15):   
                        f.write('W-s-'+str(i+1)+'='+self.rec_result +"\n")
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('W Checking NO'+'\n')         
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)     
            self.result_num.clear() 
            num=0
            
            #2.5kg称量检定
            self.zaux.send_Data(0,"\"F0300\"")#前置偏载砝码下降
            time.sleep(23)
            #主砝码加载到2.5kg(一块主砝码子块)
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_2) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('W-2.5kg-'+str(num))
            time.sleep(2)
            f.write('W-2.5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('W-2.5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('W-2.5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('W-2.5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23) 
            
            #5kg检定
            #主砝码加载到5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_5) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('W-5kg-'+str(num))
            time.sleep(2)
            f.write('W-5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('W-5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('W-5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('W-5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)     
            
            #7.5kg检定
            #主砝码加载到7.5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_7) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('W-7.5kg-'+str(num))
            time.sleep(2)
            f.write('W-7.5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('W-7.5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('W-7.5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('W-7.5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)     
            
            #10kg检定
            #主砝码加载到10kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_10) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('W-10kg-'+str(num))
            time.sleep(2)
            f.write('W-10kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('W-10kg-s-'+str(i+1))
                time.sleep(2)
                f.write('W-10kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('W-10kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)
            
            #15kg检定
            #主砝码加载到15kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_15) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('W-15kg-'+str(num))
            time.sleep(2)
            f.write('W-15kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('W-15kg-s-'+str(i+1))
                time.sleep(2)
                f.write('W-15kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('W-15kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)    
            
            #10kg检定(减载)
            #主砝码加载到15kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_10) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('W-10kg-'+str(num))
            time.sleep(2)
            f.write('W-10kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('W-10kg-s-'+str(i+1))
                time.sleep(2)
                f.write('W-10kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('W-10kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)            
            
            #7.5kg检定(减载)
            #主砝码加载到7.5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_7) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('W-7.5kg-'+str(num))
            time.sleep(2)
            f.write('W-7.5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('W-7.5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('W-7.5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('W-7.5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)  
            
            #5kg检定(减载)
            #主砝码加载到5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_5) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('W-5kg-'+str(num))
            time.sleep(2)
            f.write('W-5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('W-5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('W-5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('W-5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23) 
            
            #2.5kg检定(减载)
            #主砝码加载到2.5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_2) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('W-2.5kg-'+str(num))
            time.sleep(2)
            f.write('W-2.5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('W-2.5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('W-2.5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('W-2.5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)               
            self.zaux.send_Data(0,"\"F0201\"")#置零砝码上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0101\"")#砝码托盘上升
            time.sleep(23)
            #主砝码卸载
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_down) #加载平台下降, 辅助砝码同步下降
            time.sleep(120)
            num=0
            self.signal.emit('04-称量性能检定完成')
            
            #除皮误差性能检定
            f.write('************Skin Removal Error Performance Verification************'+"\n") #记录砝码托盘下降示值
            self.signal.emit('05-除皮误差性能检定开始')
            self.zaux.send_Data(0,"\"F0402\"")#除皮2砝码下降
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0600\"")#除皮功能按键
            time.sleep(23)
            
            self.zaux.send_Data(0, "\"F0100\"")#砝码托盘下降
            time.sleep(8) 
            self.SaveImage('SREP-'+str(num))
            time.sleep(2)
            f.write('SREP-'+str(num)+'='+self.rec_result +"\n") #记录砝码托盘下降示值
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('SREP-s-'+str(i+1))
                time.sleep(2)
                f.write('SREP-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('SREP Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)               
            
            #2.5kg除皮误差性能检定
            self.zaux.send_Data(0,"\"F0301\"")#前置偏载砝码下降
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
            time.sleep(23)
            self.SaveImage('SREP-'+str(num))
            time.sleep(2)
            f.write('SREP-'+str(num)+'='+self.rec_result +"\n") #记录砝码托盘下降示值
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('SREP-s-'+str(i+1))
                time.sleep(2)
                f.write('SREP-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('SREP Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
            time.sleep(23)
            
            #主砝码加载到2.5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_2) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('SREP-2.5kg-'+str(num))
            time.sleep(2)
            f.write('SREP-2.5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('SREP-2.5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('SREP-2.5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('SREP-2.5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)   
            
            #主砝码加载到7.5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_7) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('SREP-7.5kg-'+str(num))
            time.sleep(2)
            f.write('SREP-7.5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('SREP-7.5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('SREP-7.5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('SREP-7.5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)   
            
            #主砝码加载到10kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_10) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('SREP-10kg-'+str(num))
            time.sleep(2)
            f.write('SREP-10kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('SREP-10kg-s-'+str(i+1))
                time.sleep(2)
                f.write('SREP-10kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('SREP-10kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)      
            
            #主砝码卸载到7.5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_7) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('SREP-7.5kg-'+str(num))
            time.sleep(2)
            f.write('SREP-7.5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('SREP-7.5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('SREP-7.5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('SREP-7.5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23) 
            
            #主砝码卸载到5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_5) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('SREP-5kg-'+str(num))
            time.sleep(2)
            f.write('SREP-5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('SREP-5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('SREP-5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('SREP-5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)                                                     
           
           #主砝码卸载到2.5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_2) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('SREP-2.5kg-'+str(num))
            time.sleep(2)
            f.write('SREP-2.5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('SREP-2.5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('SREP-2.5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('SREP-2.5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23) 
            
            #主砝码卸载到0kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_down) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('SREP-0kg-'+str(num))
            time.sleep(2)
            f.write('SREP-0kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('SREP-0kg-s-'+str(i+1))
                time.sleep(2)
                f.write('SREP-0kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('SREP-0kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            self.zaux.send_Data(0, "\"F0000\"")#全部上升
            time.sleep(23)   
            num=0
            self.signal.emit('05-除皮误差性能检定完成')                    
           
           #重复性能检定
            f.write('************Repeatability Check Verification************'+"\n") 
            self.signal.emit('07-重复性能检定开始') 
            for j in range(1, 4):  #重复检定3次
                self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_15) #加载平台上升, 辅助砝码同步上升
                time.sleep(120)
                self.SaveImage('RC_'+str(j)+'-'+str(num))
                time.sleep(2)
                f.write('RC_'+str(j)+'-'+str(num)+'='+self.rec_result +"\n")
                num+=1
                self.zaux.send_Data(0,"\"F0100\"")#砝码托盘下降
                time.sleep(23)
                self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
                time.sleep(23)
                self.zaux.send_Data(0,"\"F0301\"")#前置偏载砝码下降
                time.sleep(23)
                f.write('RC_'+str(j)+'-'+str(num)+'='+self.rec_result +"\n")
                num+=1
                for i in range(15): #闪变砝码下降
                    self.zaux.send_Data(0, array_Instruction[i])
                    time.sleep(13)
                    #采集图像
                    self.SaveImage('RC_'+str(j)+'-s-'+str(i+1))
                    time.sleep(2)
                    f.write('RC_'+str(j)+'-s-'+ str(i+1) +'='+self.rec_result +"\n")
                    if self.rec_result not in self.result_num and i!=0:
                        f.write('Change_'+str(j)+'='+str(i+1))
                        break
                    else:
                        if((i+1)!=15):   
                            self.result_num.append(self.rec_result)
                        else:
                            f.write('RC'+str(j)+ 'Checking NO'+'\n')     
                self.result_num.clear() #清空
                self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
                self.zaux.send_Data(0, "\"F0201\"")#置零砝码上升
                self.zaux.send_Data(0, "\"F0301\"")#砝码托盘上升
                self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_down) #加载平台上升, 辅助砝码同步上升
                time.sleep(120)
                num=0
            self.signal.emit('07-重复性能检定完成')
            
            #鉴别阈检定
            f.write('************Identification Threshold Check Verification************'+"\n") 
            self.signal.emit('07-鉴别阈检定开始') 
            #50g除皮置零检定
            self.zaux.send_Data(0,"\"F0100\"")#砝码托盘下降
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0504\"")#00~09闪变砝码下落
            time.sleep(23)
            self.SaveImage('ITC-'+str(num))
            time.sleep(2)
            f.write('ITC-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(10): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction_2[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('ITC-s-'+str(i+1))
                time.sleep(2)
                f.write('ITC-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('ITC Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0,"\"F0AN7\"")#最后一个闪变砝码下落
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0500\"")#鉴别阈砝码下落
            time.sleep(23)
            self.SaveImage('ITC-'+str(num))
            time.sleep(2)
            f.write('ITC-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            self.zaux.send_Data(0,"\"F0500\"")#鉴别阈砝码上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
            time.sleep(23)
            #7.5kg检定
            self.zaux.send_Data(0,"\"F0301\"")#前置偏载砝码下降
            time.sleep(23)
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_7) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.zaux.send_Data(0,"\"F0504\"")#00~09闪变砝码下落
            time.sleep(23)
            self.SaveImage('ITC-'+str(num))
            time.sleep(2)
            f.write('ITC-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(10): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction_2[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('ITC-s-'+str(i+1))
                time.sleep(2)
                f.write('ITC-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('ITC Checking NO'+'\n')               
            self.zaux.send_Data(0,"\"F0AN7\"")#最后一个闪变砝码下落
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0500\"")#鉴别阈砝码下落
            time.sleep(23)
            self.SaveImage('ITC-'+str(num))
            time.sleep(2)
            f.write('ITC-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            self.zaux.send_Data(0,"\"F0500\"")#鉴别阈砝码上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
            time.sleep(23)
            #15kg检定
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_15) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.zaux.send_Data(0,"\"F0504\"")#00~09闪变砝码下落
            time.sleep(23)
            self.SaveImage('ITC-'+str(num))
            time.sleep(2)
            f.write('ITC-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(10): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction_2[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('ITC-s-'+str(i+1))
                time.sleep(2)
                f.write('ITC-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('ITC Checking NO'+'\n')      
            self.zaux.send_Data(0,"\"F0AN7\"")#最后一个闪变砝码下落
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0500\"")#鉴别阈砝码下落
            time.sleep(23)
            self.SaveImage('ITC-'+str(num))
            time.sleep(2)
            f.write('ITC-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            self.zaux.send_Data(0,"\"F0500\"")#鉴别阈砝码上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0000\"")#复位
            time.sleep(23)
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_down) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.signal.emit('07-鉴别阈检定完成') 

    #30kg性能检定       
    def Verification_experiment_30kg(self):
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
            #轴运动参数设置
            #轴号列表
            Z1_Axis_List=[0, 1] 
            Z_Axis_List=[0, 1, 2]  
            YX_Axis_list=[4, 3] #YX轴号列表
            ZYX_Axis_list=[0, 1, 2, 4, 3] #YX轴号列表
            #运动位置
            Z1_Data_list=[112, 112] #Z1轴上升距离
            Z_Data_list_up_2=[Z1_Data_list[0]+120+5, Z1_Data_list[1]+120+5, 120+5] #2.5kg加载距离
            Z_Data_list_up_5=[Z1_Data_list[0]+120+10, Z1_Data_list[1]+120+10, 120+10] #5kg加载距离
            Z_Data_list_up_7=[Z1_Data_list[0]+120+15, Z1_Data_list[1]+120+15, 120+15] #7.5kg加载距离
            Z_Data_list_up_10=[Z1_Data_list[0]+120+20, Z1_Data_list[1]+120+20, 120+20] #10kg加载距离
            Z_Data_list_up_15=[Z1_Data_list[0]+120+25, Z1_Data_list[1]+120+25, 120+25] #15kg加载距离
            Z_Data_list_up_15=[Z1_Data_list[0]+120+30, Z1_Data_list[1]+120+30, 120+30] #30kg加载距离
            Z_Data_list_down=[Z1_Data_list[0], Z1_Data_list[1], 0]
            
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
            
            array_Instruction_2=[ "\"F0BN9\"" ,"\"F0BN8\"","\"F0BN7\"",
                                            "\"F0BN6\"","\"F0BN5\"","\"F0BN4\"",
                                            "\"F0BN3\"","\"F0BN2\"","\"F0BN1\"",
                                            "\"F0BN0\""]
            self.zaux.setCom_defaultBaud(9600, 8, 1, 0) #开启串口通信     
            f.write('Verification_experiment_30kg'+"\n")
            self.signal.emit('30kg量程电子秤检定实验')
            #30kg量程-置零性检定
            f.write('************Zero Verification************'+"\n") #记录砝码托盘下降示值
            self.signal.emit('01-置零检定开始')
            self.zaux.multiAxis_moveAbs(2, Z1_Axis_List, Z1_Data_list) #ZI1轴平台上升
            time.sleep(80)
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
            self.signal.emit('01-置零检定已完成')
            num=0
    
            #30kg量程-偏载性检定
            f.write('\n************Off-load calibration************'+"\n") #
            self.signal.emit('02-偏载检定开始')
            #偏载性能左上角检定
            f.write('************Left_Top************'+"\n") 
            self.signal.emit('02-偏载检定-左上角')
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
            time.sleep(110)
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
            time.sleep(110)
            num=0
            ##偏载性能右上角检定
            f.write('************Right Top************'+"\n") 
            self.signal.emit('02-偏载检定-右上角')
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
            self.signal.emit('02-偏载检定-右下角')
            f.write('************Right Bottom************'+"\n") 
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
            #主砝码卸载
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_down) #加载平台下降, 辅助砝码同步下降
            time.sleep(120)
            num=0
            
            # 偏载性能左下角检定
            self.signal.emit('02-偏载检定-左下角')
            f.write('************Left Bottom************'+"\n") 
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
                                    
            #偏载性能中心位置检定
            self.signal.emit('02-偏载检定-中心位置')
            f.write('************Center************'+"\n") 
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
            self.zaux.send_Data(0,"\"F0303\"")#后置置砝码上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0101\"")#砝码托盘上升
            time.sleep(23)
            #主砝码卸载(一块主砝码子块
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_down) #加载平台下降, 辅助砝码同步下降
            time.sleep(120)
            num=0
            self.signal.emit('02-偏载检定完成')
            
            #置零准确度检定
            f.write('************Zero Verification************'+"\n") #记录砝码托盘下降示值
            self.signal.emit('03-置零检定开始')
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
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码抬升
            time.sleep(23)     
            self.result_num.clear() 
            self.signal.emit('03-置零检定已完成')
            num=0
            
            #称量性能检定
            f.write('************Weighing Performance Verification************'+"\n") #记录砝码托盘下降示值
            self.signal.emit('04-称量性能检定开始')
            self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
            time.sleep(23)
            self.SaveImage('W-'+str(num))
            f.write('W-'+str(num)+'='+self.rec_result +"\n") #记录置零砝码下落示值
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                self.SaveImage('W-'+str(num)) #采集图像
                if self.rec_result not in self.result_num and i!=0:
                    f.write('W-s-'+str(i+1)+'='+self.rec_result +"\n")
                    f.write('W Change='+str(i+1)+'\n')
                    break
                else:
                    if((i+1)!=15):   
                        f.write('W-s-'+str(i+1)+'='+self.rec_result +"\n")
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('W Checking NO'+'\n')         
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)     
            self.result_num.clear() 
            num=0
            
            #2.5kg称量检定
            self.zaux.send_Data(0,"\"F0300\"")#前置偏载砝码下降
            time.sleep(23)
            #主砝码加载到2.5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_2) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('W-2.5kg-'+str(num))
            time.sleep(2)
            f.write('W-2.5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('W-2.5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('W-2.5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('W-2.5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23) 
            
            #5kg检定
            #主砝码加载到5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_5) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('W-5kg-'+str(num))
            time.sleep(2)
            f.write('W-5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('W-5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('W-5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('W-5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)     
            
            #7.5kg检定
            #主砝码加载到7.5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_7) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('W-7.5kg-'+str(num))
            time.sleep(2)
            f.write('W-7.5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('W-7.5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('W-7.5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('W-7.5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)     
            
            #10kg检定
            #主砝码加载到10kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_10) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('W-10kg-'+str(num))
            time.sleep(2)
            f.write('W-10kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('W-10kg-s-'+str(i+1))
                time.sleep(2)
                f.write('W-10kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('W-10kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)
            
            #15kg检定
            #主砝码加载到15kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_15) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('W-15kg-'+str(num))
            time.sleep(2)
            f.write('W-15kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('W-15kg-s-'+str(i+1))
                time.sleep(2)
                f.write('W-15kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('W-15kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)    
            
            #10kg检定(减载)
            #主砝码加载到15kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_10) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('W-10kg-'+str(num))
            time.sleep(2)
            f.write('W-10kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('W-10kg-s-'+str(i+1))
                time.sleep(2)
                f.write('W-10kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('W-10kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)            
            
            #7.5kg检定(减载)
            #主砝码加载到7.5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_7) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('W-7.5kg-'+str(num))
            time.sleep(2)
            f.write('W-7.5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('W-7.5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('W-7.5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('W-7.5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)  
            
            #5kg检定(减载)
            #主砝码加载到5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_5) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('W-5kg-'+str(num))
            time.sleep(2)
            f.write('W-5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('W-5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('W-5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('W-5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23) 
            
            #2.5kg检定(减载)
            #主砝码加载到2.5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_2) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('W-2.5kg-'+str(num))
            time.sleep(2)
            f.write('W-2.5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('W-2.5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('W-2.5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('W-2.5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)               
            self.zaux.send_Data(0,"\"F0201\"")#置零砝码上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0101\"")#砝码托盘上升
            time.sleep(23)
            #主砝码卸载
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_down) #加载平台下降, 辅助砝码同步下降
            time.sleep(120)
            num=0
            self.signal.emit('04-称量性能检定完成')
            
            #除皮误差性能检定
            f.write('************Skin Removal Error Performance Verification************'+"\n") #记录砝码托盘下降示值
            self.signal.emit('05-除皮误差性能检定开始')
            self.zaux.send_Data(0,"\"F0402\"")#除皮2砝码下降
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0600\"")#除皮功能按键
            time.sleep(23)
            
            self.zaux.send_Data(0, "\"F0100\"")#砝码托盘下降
            time.sleep(8) 
            self.SaveImage('SREP-'+str(num))
            time.sleep(2)
            f.write('SREP-'+str(num)+'='+self.rec_result +"\n") #记录砝码托盘下降示值
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('SREP-s-'+str(i+1))
                time.sleep(2)
                f.write('SREP-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('SREP Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)               
            
            #2.5kg除皮误差性能检定
            self.zaux.send_Data(0,"\"F0301\"")#前置偏载砝码下降
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
            time.sleep(23)
            self.SaveImage('SREP-'+str(num))
            time.sleep(2)
            f.write('SREP-'+str(num)+'='+self.rec_result +"\n") #记录砝码托盘下降示值
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('SREP-s-'+str(i+1))
                time.sleep(2)
                f.write('SREP-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('SREP Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
            time.sleep(23)
            
            #主砝码加载到2.5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_2) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('SREP-2.5kg-'+str(num))
            time.sleep(2)
            f.write('SREP-2.5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('SREP-2.5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('SREP-2.5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('SREP-2.5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)   
            
            #主砝码加载到7.5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_7) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('SREP-7.5kg-'+str(num))
            time.sleep(2)
            f.write('SREP-7.5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('SREP-7.5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('SREP-7.5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('SREP-7.5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)   
            
            #主砝码加载到10kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_10) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('SREP-10kg-'+str(num))
            time.sleep(2)
            f.write('SREP-10kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('SREP-10kg-s-'+str(i+1))
                time.sleep(2)
                f.write('SREP-10kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('SREP-10kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)      
            
            #主砝码卸载到7.5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_7) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('SREP-7.5kg-'+str(num))
            time.sleep(2)
            f.write('SREP-7.5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('SREP-7.5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('SREP-7.5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('SREP-7.5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23) 
            
            #主砝码卸载到5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_5) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('SREP-5kg-'+str(num))
            time.sleep(2)
            f.write('SREP-5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('SREP-5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('SREP-5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('SREP-5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23)                                                     
           
           #主砝码卸载到2.5kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_2) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('SREP-2.5kg-'+str(num))
            time.sleep(2)
            f.write('SREP-2.5kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('SREP-2.5kg-s-'+str(i+1))
                time.sleep(2)
                f.write('SREP-2.5kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('SREP-2.5kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            time.sleep(23) 
            
            #主砝码卸载到0kg
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_down) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.SaveImage('SREP-0kg-'+str(num))
            time.sleep(2)
            f.write('SREP-0kg-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(15): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('SREP-0kg-s-'+str(i+1))
                time.sleep(2)
                f.write('SREP-0kg-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('SREP-0kg Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
            self.zaux.send_Data(0, "\"F0000\"")#全部上升
            time.sleep(23)   
            num=0
            self.signal.emit('05-除皮误差性能检定完成')                    
           
           #重复性能检定
            f.write('************Repeatability Check Verification************'+"\n") 
            self.signal.emit('07-重复性能检定开始') 
            for j in range(1, 4):  #重复检定3次
                self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_15) #加载平台上升, 辅助砝码同步上升
                time.sleep(120)
                self.SaveImage('RC_'+str(j)+'-'+str(num))
                time.sleep(2)
                f.write('RC_'+str(j)+'-'+str(num)+'='+self.rec_result +"\n")
                num+=1
                self.zaux.send_Data(0,"\"F0100\"")#砝码托盘下降
                time.sleep(23)
                self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
                time.sleep(23)
                self.zaux.send_Data(0,"\"F0301\"")#前置偏载砝码下降
                time.sleep(23)
                f.write('RC_'+str(j)+'-'+str(num)+'='+self.rec_result +"\n")
                num+=1
                for i in range(15): #闪变砝码下降
                    self.zaux.send_Data(0, array_Instruction[i])
                    time.sleep(13)
                    #采集图像
                    self.SaveImage('RC_'+str(j)+'-s-'+str(i+1))
                    time.sleep(2)
                    f.write('RC_'+str(j)+'-s-'+ str(i+1) +'='+self.rec_result +"\n")
                    if self.rec_result not in self.result_num and i!=0:
                        f.write('Change_'+str(j)+'='+str(i+1))
                        break
                    else:
                        if((i+1)!=15):   
                            self.result_num.append(self.rec_result)
                        else:
                            f.write('RC'+str(j)+ 'Checking NO'+'\n')     
                self.result_num.clear() #清空
                self.zaux.send_Data(0, "\"F00U0\"")#闪变砝码全部抬升
                self.zaux.send_Data(0, "\"F0201\"")#置零砝码上升
                self.zaux.send_Data(0, "\"F0301\"")#砝码托盘上升
                self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_down) #加载平台上升, 辅助砝码同步上升
                time.sleep(120)
                num=0
            self.signal.emit('07-重复性能检定完成')
            
            #鉴别阈检定
            f.write('************Identification Threshold Check Verification************'+"\n") 
            self.signal.emit('07-鉴别阈检定开始') 
            #50g除皮置零检定
            self.zaux.send_Data(0,"\"F0100\"")#砝码托盘下降
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0200\"")#置零砝码下降
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0504\"")#00~09闪变砝码下落
            time.sleep(23)
            self.SaveImage('ITC-'+str(num))
            time.sleep(2)
            f.write('ITC-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(10): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction_2[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('ITC-s-'+str(i+1))
                time.sleep(2)
                f.write('ITC-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('ITC Checking NO'+'\n')                 
            self.result_num.clear() #清空
            self.zaux.send_Data(0,"\"F0AN7\"")#最后一个闪变砝码下落
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0500\"")#鉴别阈砝码下落
            time.sleep(23)
            self.SaveImage('ITC-'+str(num))
            time.sleep(2)
            f.write('ITC-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            self.zaux.send_Data(0,"\"F0500\"")#鉴别阈砝码上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
            time.sleep(23)
            #7.5kg检定
            self.zaux.send_Data(0,"\"F0301\"")#前置偏载砝码下降
            time.sleep(23)
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_7) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.zaux.send_Data(0,"\"F0504\"")#00~09闪变砝码下落
            time.sleep(23)
            self.SaveImage('ITC-'+str(num))
            time.sleep(2)
            f.write('ITC-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(10): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction_2[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('ITC-s-'+str(i+1))
                time.sleep(2)
                f.write('ITC-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('ITC Checking NO'+'\n')               
            self.zaux.send_Data(0,"\"F0AN7\"")#最后一个闪变砝码下落
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0500\"")#鉴别阈砝码下落
            time.sleep(23)
            self.SaveImage('ITC-'+str(num))
            time.sleep(2)
            f.write('ITC-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            self.zaux.send_Data(0,"\"F0500\"")#鉴别阈砝码上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
            time.sleep(23)
            #15kg检定
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_up_15) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.zaux.send_Data(0,"\"F0504\"")#00~09闪变砝码下落
            time.sleep(23)
            self.SaveImage('ITC-'+str(num))
            time.sleep(2)
            f.write('ITC-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            for i in range(10): #闪变砝码下降
                self.zaux.send_Data(0, array_Instruction_2[i])
                time.sleep(13)
                #采集图像
                self.SaveImage('ITC-s-'+str(i+1))
                time.sleep(2)
                f.write('ITC-s-'+ str(i+1) +'='+self.rec_result +"\n")
                if self.rec_result not in self.result_num and i!=0:
                    f.write('Change='+str(i+1))
                    break
                else:
                    if((i+1)!=15):   
                        self.result_num.append(self.rec_result)
                    else:
                        f.write('ITC Checking NO'+'\n')      
            self.zaux.send_Data(0,"\"F0AN7\"")#最后一个闪变砝码下落
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0500\"")#鉴别阈砝码下落
            time.sleep(23)
            self.SaveImage('ITC-'+str(num))
            time.sleep(2)
            f.write('ITC-'+str(num)+'='+self.rec_result +"\n")
            num+=1
            self.zaux.send_Data(0,"\"F0500\"")#鉴别阈砝码上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F00U0\"")#闪变砝码全部上升
            time.sleep(23)
            self.zaux.send_Data(0,"\"F0000\"")#复位
            time.sleep(23)
            self.zaux.multiAxis_moveAbs(3, Z_Axis_List, Z_Data_list_down) #加载平台上升, 辅助砝码同步上升
            time.sleep(120)
            self.signal.emit('07-鉴别阈检定完成') 
    
    def run(self):
        # 进行多任务操作
        if self.status == "connected":  
            self.connected()   
        elif self.status == "openCam":
             self.openCam()             
        elif self.status == "test":
            self.test()
        elif self.status == "model_Load":
            self.model_Load()
        elif self.status == "Verification_experiment_15kg":
            self.Verification_experiment_15kg()
        elif self.status == "Verification_experiment_30kg":
            self.Verification_experiment_30kg()
        elif self.status == "restart":
            self.restart()
        elif self.status == "closeCam":
            self.closeCam()
        else:
            pass
        # 发射信号 
        self.signal.emit(self.message)
        self.image_signal.emit(self.image_name) 
        