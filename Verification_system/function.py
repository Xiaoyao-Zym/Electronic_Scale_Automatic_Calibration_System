import platform, os
import ctypes
#导入库

systype = platform.system()
if systype == 'Windows':
    if platform.architecture()[0]!= '64bit':
        zauxdll = ctypes.WinDLL('./zauxdll64.dll')
        print('Windows x64')
    else:
        path=os.path.join(os.getcwd(), 'Verification_system\\zauxdll.dll')
        zauxdll = ctypes.WinDLL(path)
        print('Windows x86')
elif systype == 'Darwin':
    zmcdll = ctypes.CDLL('./zmotion.dylib')
    print("macOS")
elif systype == 'Linux':
    zmcdll = ctypes.CDLL('./libbzmotion.so')
    print("Linux")
else:
    print("OS Not Supported!!")


class ZMCWrapper:

    def __init__(self):
        self.handle = ctypes.c_void_p()
        self.sys_ip = ""
        self.sys_info = ""
        self.is_connected = False
        #self.connect(ip)
        
    #连接控制器 
    def connect(self, ip, console=[]):
        if self.handle.value is not None:
            self.disconnect()
        ip_bytes = ip.encode('utf-8')
        p_ip = ctypes.c_char_p(ip_bytes)
        #print("Connecting to", ip, "...")
        ret = zauxdll.ZAux_OpenEth(p_ip, ctypes.pointer(self.handle))
        msg = "Connected"
        if ret == 0:
            msg = ip + "Already Connected"
            self.sys_ip = ip
            self.is_connected = True
        else:
            msg = "Connection Failed, Error " + str(ret)
            self.is_connected = False
        console.append(msg)
        console.append(self.sys_info)
        return ret
    
    #断开控制器连接
    def disconnect(self):
        ret = zauxdll.ZAux_Close(self.handle)
        self.is_connected = False
        return ret
    
    #设置轴类型
    def set_atype(self, iaxis, iValue):
        ret = zauxdll.ZAux_Direct_SetAtype(self.handle, iaxis, iValue)
        if ret == 0:
            print("Set Axis (", iaxis, ") Atype:", iValue)
        else:
            print("Set Axis (", iaxis, ") Atype fail!")
        return ret
    
    #设置轴脉冲当量
    def set_units(self, iaxis, iValue):
        ret = zauxdll.ZAux_Direct_SetUnits(self.handle, iaxis, ctypes.c_float(iValue))
        if ret == 0:
            print("Set Axis (", iaxis, ") Units:", iValue)
        else:
            print("Set Axis (", iaxis, ") Units fail!")
        return ret

    #设置轴加速度
    def set_accel(self, iaxis, iValue):
        ret = zauxdll.ZAux_Direct_SetAccel(self.handle, iaxis, ctypes.c_float(iValue))
        if ret == 0:
            print("Set Axis (", iaxis, ") Accel:", iValue)
        else:
            print("Set Accel (", iaxis, ") Accel fail!")
        return ret
    
    #设置轴减速度
    def set_decel(self, iaxis, iValue):
        ret = zauxdll.ZAux_Direct_SetDecel(self.handle, iaxis, ctypes.c_float(iValue))
        if ret == 0:
            print("Set Axis (", iaxis, ") Decel:", iValue)
        else:
            print("Set Axis (", iaxis, ") Decel fail!")
        return ret
    
    #设置轴运行速度
    def set_speed(self, iaxis, iValue):
        ret = zauxdll.ZAux_Direct_SetSpeed(self.handle, iaxis, ctypes.c_float(iValue))
        if ret == 0:
            print("Set Axis (", iaxis, ") Speed:", iValue)
        else:
            print("Set Axis (", iaxis, ") Speed fail!")
        return ret

    #获取轴类型
    def get_atype(self, iaxis):
        iValue = (ctypes.c_int)()
        ret = zauxdll.ZAux_Direct_GetAtype(self.handle, iaxis, ctypes.byref(iValue))
        if ret == 0:
            print("Get Axis (", iaxis, ") Atype:", iValue.value)
        else:
            print("Get Axis (", iaxis, ") Atype fail!")
        return ret

    #获取轴脉冲当量
    def get_untis(self, iaxis):
        iValue = (ctypes.c_float)()
        ret = zauxdll.ZAux_Direct_GetUnits(self.handle, iaxis, ctypes.byref(iValue))
        if ret == 0:
            print("Get Axis (", iaxis, ") Units:", iValue.value)
        else:
            print("Get Axis (", iaxis, ") Units fail!")
        return ret
    
   #读取轴加速度
    def get_accel(self, iaxis):
        iValue = (ctypes.c_float)()
        ret = zauxdll.ZAux_Direct_GetAccel(self.handle, iaxis, ctypes.byref(iValue))
        if ret == 0:
            print("Get Axis (", iaxis, ") Accel:",  iValue.value)
        else:
            print("Get Axis (", iaxis, ") Accel fail!")
        return ret
    
   #读取轴减速度
    def get_decel(self, iaxis):
        iValue = (ctypes.c_float)()
        ret = zauxdll.ZAux_Direct_GetDecel(self.handle, iaxis, ctypes.byref(iValue))
        if ret == 0:
            print("Get Axis (", iaxis, ") Decel:",  iValue.value)
        else:
            print("Get Axis (", iaxis, ") Decel fail!")
        return ret

    # 读取轴运行速度
    def get_speed(self, iaxis):
        iValue = (ctypes.c_float)()
        ret = zauxdll.ZAux_Direct_GetSpeed(self.handle, iaxis, ctypes.byref(iValue))
        if ret == 0:
            print("Get Axis (", iaxis, ") Speed:",  iValue.value)
        else:
            print("Get Axis (", iaxis, ") Speed fail!")
        return ret
    
    #####运动调用####
     #单轴相对距离运动
    def move(self, iaxis, iValue):
        ret = zauxdll.ZAux_Direct_Single_Move(self.handle, iaxis, ctypes.c_float(iValue))
        if ret == 0:
            print("Axis (", iaxis, ") Move:", iValue)
        else:
            print("Axis (", iaxis, ") Move Fail")
        return ret

    #单轴连续运动
    def vmove(self, iaxis, idir):
        ret = zauxdll.ZAux_Direct_Single_Vmove(self.handle, iaxis, idir)
        if ret == 0:
            print("axis (", iaxis, ")Vmoving!")
        else:
            print("Vmoving fail!")
        return ret
    
    #单轴绝对距离运动
    def singleAxis_moveAbs(self, iaxis,  iValue):
        ret = zauxdll.ZAux_Direct_Single_MoveAbs(self.handle, iaxis, ctypes.c_float(iValue))
        if ret == 0:
            print("axis (", iaxis, ")Vmoving!", iValue)
        else:
            print("Vmoving fail!")
        return ret
    
    #多轴相对直线插补运动
    def multiAxis_move(self, imaxaxises, iaxis, iValue):
        ret = zauxdll.ZAux_Direct_Move(self.handle, imaxaxises,  iaxis,  iValue)
        if ret == 0:
            print("Axis (", iaxis, ") multi_Move:", iValue)
        else:
            print("Axis (", iaxis, ") multi_Move Fail")
        return ret
    
    #多轴绝对直线插补运动
    def multiAxis_moveAbs(self, imaxaxises, iaxis, iValue):
        ret = zauxdll.ZAux_Direct_MoveAbs(self.handle, imaxaxises, (ctypes.c_int*len(iaxis))(*iaxis),  (ctypes.c_float * len(iValue))(*iValue))
        if ret == 0:
            print("Axis (", iaxis, ") multi_Move:", iValue)
        else:
            print("Axis (", iaxis, ") multi_Move Fail")
        return ret
    
    #设置串口通讯
    def setCom_defaultBaud(self, dwBaudRate, dwByteSize, dwParity, dwStopBits):
        cmdbuffAck=(ctypes.c_char*2048)()
        cmdbuff='SETCOM(%d, %d, %d, %d, 0, 0, 0)'%(dwBaudRate, dwByteSize,  dwParity,  dwStopBits)
        cmd=ctypes.c_char_p(bytes(cmdbuff, 'utf-8'))
        ret = zauxdll.ZAux_Execute(self.handle, cmd, cmdbuffAck, 2048)
        if ret == 0:
            print("setCom_defaultBaud:", dwBaudRate, dwByteSize,  dwParity,  dwStopBits)
        else:
            print("setCom_defaultBaud Fail")
        return ret
        
    #发送数据函数
    def send_Data(self, port, command):
        cmdbuffAck=(ctypes.c_char*2048)()
        cmdbuff='PRINT #%d,%s'%(port, command)
        cmd=ctypes.c_char_p(bytes(cmdbuff, 'utf-8'))
        ret = zauxdll.ZAux_Execute(self.handle, cmd, cmdbuffAck, 2048)
        if ret == 0:
            print(cmdbuff)
        else:
            print("command (", command, ") send_Data Fail")
        return ret

        
        
        





