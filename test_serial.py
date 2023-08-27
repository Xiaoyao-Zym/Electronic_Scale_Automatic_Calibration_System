import serial
def send_receive_strings(port, baudrate, send_string, num_strings):
    # 打开串口
    ser = serial.Serial(port, baudrate, timeout=1)
    # 发送字符串
    ser.write(send_string.encode())
    if send_string=='FZERO':
        # 接收字符串
        received_strings = []
        while len(received_strings) < num_strings:
            received_string = ser.readline().decode().strip()
            if received_string:
                received_strings.append(received_string)
        return int(received_string.lstrip("0"))
    else:
        pass

if __name__ == "__main__":
    print(send_receive_strings('COM8', 9600, 'FZERO', 2))
