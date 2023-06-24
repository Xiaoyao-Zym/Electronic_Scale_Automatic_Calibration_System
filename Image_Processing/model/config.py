 # 模型相关参数配置
import argparse
def config_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_model_path', type=str, default='../weights/yolox_d.pt', help='model.pt path(s)')
    parser.add_argument('--rec_model_path', type=str, default='../weights/drnet_v3.pt', help='model.pt path(s)')
    parser.add_argument('--classes_path', type=str, default='./classes.txt', help='class list')  # file/folder, 0 for webcam
    parser.add_argument('--input_shape', type=list, default=[640, 640], help='inference size (pixels)')
    parser.add_argument('--phi', type=str, default='s', help='#   所使用的YoloX的版本。nano、tiny、s、m、l、x')
    parser.add_argument('--confidence', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--cuda', type=bool, default=True, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--letterbox_image', type=bool, default= True, help='display results')
    parser.add_argument('--nms_iou', type=float, default=0.3, help='非极大抑制所用到的nms_iou大小')
    opt = parser.parse_args()
    return opt
