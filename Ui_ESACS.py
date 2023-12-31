# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\Yanmin Zhang\Desktop\Electronic_Scale_Automatic_Calibration_System\ESACS.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1527, 903)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Camera_Connect = QtWidgets.QPushButton(self.centralwidget)
        self.Camera_Connect.setEnabled(True)
        self.Camera_Connect.setGeometry(QtCore.QRect(180, 20, 140, 50))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(14)
        self.Camera_Connect.setFont(font)
        self.Camera_Connect.setObjectName("Camera_Connect")
        self.Image_Show = QtWidgets.QLabel(self.centralwidget)
        self.Image_Show.setGeometry(QtCore.QRect(10, 90, 1191, 751))
        self.Image_Show.setFrameShape(QtWidgets.QFrame.Box)
        self.Image_Show.setText("")
        self.winid = self.Image_Show.winId()  # 获取label对象的句柄
        self.Image_Show.setObjectName("Image_Show")
        self.Camera_Gather = QtWidgets.QPushButton(self.centralwidget)
        self.Camera_Gather.setGeometry(QtCore.QRect(340, 20, 140, 50))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(14)
        self.Camera_Gather.setFont(font)
        self.Camera_Gather.setObjectName("Camera_Gather")
        self.test = QtWidgets.QPushButton(self.centralwidget)
        self.test.setGeometry(QtCore.QRect(660, 20, 161, 50))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(14)
        self.test.setFont(font)
        self.test.setObjectName("test")
        self.Camera_Close = QtWidgets.QPushButton(self.centralwidget)
        self.Camera_Close.setGeometry(QtCore.QRect(840, 20, 151, 50))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(14)
        self.Camera_Close.setFont(font)
        self.Camera_Close.setObjectName("Camera_Close")
        self.Model_Load = QtWidgets.QPushButton(self.centralwidget)
        self.Model_Load.setGeometry(QtCore.QRect(500, 20, 140, 50))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(14)
        self.Model_Load.setFont(font)
        self.Model_Load.setObjectName("Model_Load")
        self.event_message = QtWidgets.QTextEdit(self.centralwidget)
        self.event_message.setGeometry(QtCore.QRect(1240, 90, 271, 491))
        self.event_message.setObjectName("event_message")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(1200, 150, 41, 341))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(1200, 630, 41, 151))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.Controller_Experiment_15kg = QtWidgets.QPushButton(self.centralwidget)
        self.Controller_Experiment_15kg.setEnabled(True)
        self.Controller_Experiment_15kg.setGeometry(QtCore.QRect(1010, 20, 161, 50))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(14)
        self.Controller_Experiment_15kg.setFont(font)
        self.Controller_Experiment_15kg.setObjectName("Controller_Experiment_15kg")
        self.Controller_Connect = QtWidgets.QPushButton(self.centralwidget)
        self.Controller_Connect.setEnabled(True)
        self.Controller_Connect.setGeometry(QtCore.QRect(10, 20, 151, 50))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(14)
        self.Controller_Connect.setFont(font)
        self.Controller_Connect.setObjectName("Controller_Connect")
        self.restart = QtWidgets.QPushButton(self.centralwidget)
        self.restart.setGeometry(QtCore.QRect(1390, 20, 101, 50))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(14)
        self.restart.setFont(font)
        self.restart.setObjectName("restart")
        self.result_image = QtWidgets.QLabel(self.centralwidget)
        self.result_image.setGeometry(QtCore.QRect(1240, 600, 271, 241))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(14)
        self.result_image.setFont(font)
        self.result_image.setFrameShape(QtWidgets.QFrame.Box)
        self.result_image.setObjectName("result_image")
        self.Controller_Experiment_30kg = QtWidgets.QPushButton(self.centralwidget)
        self.Controller_Experiment_30kg.setEnabled(True)
        self.Controller_Experiment_30kg.setGeometry(QtCore.QRect(1190, 20, 161, 50))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(14)
        self.Controller_Experiment_30kg.setFont(font)
        self.Controller_Experiment_30kg.setObjectName("Controller_Experiment_30kg")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1527, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "电子秤自动检定系统"))
        self.Camera_Connect.setText(_translate("MainWindow", "连接相机"))
        self.Camera_Gather.setText(_translate("MainWindow", "图像采集"))
        self.test.setText(_translate("MainWindow", "测试"))
        self.Camera_Close.setText(_translate("MainWindow", "关闭相机"))
        self.Model_Load.setText(_translate("MainWindow", "模型加载"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">运</span></p><p align=\"center\"><span style=\" font-size:10pt;\">行</span></p><p align=\"center\"><span style=\" font-size:10pt;\">状</span></p><p align=\"center\"><span style=\" font-size:10pt;\">态</span></p><p align=\"center\"><span style=\" font-size:10pt;\">信</span></p><p align=\"center\"><span style=\" font-size:10pt;\">息</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">识</span></p><p align=\"center\"><span style=\" font-size:10pt;\">别</span></p><p align=\"center\"><span style=\" font-size:10pt;\">结</span></p><p align=\"center\"><span style=\" font-size:10pt;\">果</span></p></body></html>"))
        self.Controller_Experiment_15kg.setText(_translate("MainWindow", "15Kg检定"))
        self.Controller_Connect.setText(_translate("MainWindow", "连接控制器"))
        self.restart.setText(_translate("MainWindow", "复位"))
        self.result_image.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><br/></p></body></html>"))
        self.Controller_Experiment_30kg.setText(_translate("MainWindow", "30Kg检定"))
