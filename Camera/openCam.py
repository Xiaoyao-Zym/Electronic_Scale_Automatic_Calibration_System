import sys
import os
import shutil
from Camera.MVGigE import *
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QMessageBox, QLabel, QFileDialog, QScrollArea, QComboBox, QLineEdit, QSlider, QGridLayout, QGroupBox, QCheckBox
from PyQt5.QtGui import QPixmap, QPalette, QImage, QIcon
from PyQt5.QtCore import Qt

 # 点击界面打开相机按钮，执行本函数
def openCam():
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
    hCam = hCam.hCam
    width = w.width
    height = h.height
    pixelFormat = pf.pixelFormat
    if(pixelFormat == MV_PixelFormatEnums.PixelFormat_Mono8):
        himage = MVImageCreate(width, height, 8).himage  # 创建图像句柄
    else:
        himage = MVImageCreate(width, height, 24).himage  # 创建图像句柄