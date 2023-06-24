# -*-coding: utf-8 -*-
 
import os, sys
from PIL import Image  
sys.path.append(os.getcwd())
import onnxruntime
import onnx
from PIL import ImageDraw, ImageFont
 
 
import cv2
import numpy as np
import onnxruntime as rt

import cv2
import numpy as np
import time
import onnxruntime
import onnx
import onnxruntime as rt
from yolox_onnx import YOLOX_ONNX

import torch


def decode_outputs(outputs, input_shape):
    """

    Args:
        outputs:列表，里面的元素分别为各个检测头的输出
        input_shape:列表或元组，里面的两个元素分别为模型输入图片的高宽，如[640, 640]

    Returns:

    """

    """以下代码的注释，都是假设只有三个检测头，要检测的类别数是80，input_shape为[640, 640]的情况下的结果"""
    grids = []
    strides = []

    hw = [x.shape[-2:] for x in outputs]    # 三个检测头输出结果的高宽
    
    y = []
    
    for x in range(len(outputs)):
        y.append(torch.tensor(outputs[x].reshape(-1, size[x] , 5 + class_num)).permute(0, 2, 1))
        
    outputs =torch.cat([x for x in y], dim=2)
    outputs=outputs.permute(0, 2, 1)
    
    
    # [x.flatten(start_dim=2) for x in outputs]每次获得的x都是4个维度，
    # 第一个x的维度为torch.Size([batch_size, 85, 80, 80])
    # x.flatten(start_dim=2) 表示从2号维度开始打平，打平后的维度为torch.Size([batch_size, 85, 6400])
    # 列表推导式获得的列表中，有三个张量，维度分别为(batch_size, 85, 6400)、(batch_size, 85, 1600)、(batch_size, 85, 400)
    # torch.cat将列表中的三个张量按指定维度（dim=2）拼接进行拼接，得到的张量维度为torch.Size([batch_size, 85, 8400])
    # .permute(0, 2, 1)表示调整维度顺序，得到的张量维度为torch.Size([batch_size, 8400, 85])
    # 最后的outputs的shape变为torch.Size([batch_size, 8400, 85])

    outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])
    # 最后一个维度，前面4个数是中心点坐标和高宽，从第5个数是执行都，后面是各个类别的概率，
    # 这里使用sigmoid函数将置信度和各个类别的概率压缩到0-1之间

    for h, w in hw:
        """循环中代码的注释时在第一轮循环时的结果，第一次循环，h和w是第一个特征图（dark3）的高宽，它们都是80
        第二轮循环h和w是40，第三轮循环h和w是20"""
        # 根据特征层生成网格点
        grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)])
        # grid_y和grid_x的维度都是torch.Size([80, 80])

        grid = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)
        # shape为torch.Size([1, 6400, 2])，最后一个维度是2，为网格点的横纵坐标，而6400表示当前特征层的网格点数量
        # torch.stack((grid_x, grid_y), 2)对张量进行扩维拼接，返回的shape为torch.Size([80, 80, 2])
        # 关于torch.stack的用法，可以看这篇博客：https://blog.csdn.net/Teeyohuang/article/details/80362756/

        shape = grid.shape[:2]      # shape为torch.Size([1, 6400])

        grids.append(grid)
        strides.append(torch.full((shape[0], shape[1], 1), input_shape[0] / h))
        # input_shape[0]/h 获得当前特征图（检测头对应的特征图）高h方向的步长，这个步长也是宽w方向上的步长
        # 因为因为输入图片和检测头输出的特征图，在高和宽两个方向上的缩放比例是一样的，所以步长也是一样
        # torch.full((shape[0], shape[1], 1), input_shape[0]/h是由步长填充而成的张量

    # 将网格点堆叠到一起
    grids = torch.cat(grids, dim=1).type(outputs.type())        # torch.cat是让张量按照指定维度拼接，但得到的新张量维度数不会变
    # grides的维度为(1, 8400, 2),中间的8400表示8400个特征点

    strides = torch.cat(strides, dim=1).type(outputs.type())    # .type(outputs.type())指定张量的类型
    # strides的维度为(1, 8400, 1)

    # 根据网格点进行解码
    outputs[..., :2] = (outputs[..., :2] + grids) * strides     # 解码得到中心点的坐标
    # 因为outputs[..., :2]是在0-1之间，而且其表示的中心点坐标是相对于网格点进行归一化后的，现在要将其转变成相对于整张图片
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides  # 解码得到预测框的高宽

    # 归一化（相对于图片大小）
    outputs[..., [0, 2]] = outputs[..., [0, 2]] / input_shape[1]
    outputs[..., [1, 3]] = outputs[..., [1, 3]] / input_shape[0]

    # 返回的outputs的维度为(batch_size, 8400, 85)
    return outputs

#获取边框之间的IOU
def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    Args:
        box1: 维度为(num_objects, 4)
        box2: 维度为(num_objects, 4)
        x1y1x2y2: 表示输入的目标框是否为上下角点坐标

    Returns:

    """

    # 获得边框左上角点和右下角点的坐标
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    # 计算真实框与预测框的交集矩形的左上角点和右下角点的坐标
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area交集面积
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # torch.clamp是上下限控制函数，这里使用这个函数，是因为真实框与目标框可能不存在交集
    # 那么inter_rect_x2-inter_rect_x1+1 或者 inter_rect_y2 - inter_rect_y1+1 就是负的
    # TODO 这里inter_rect_x2-inter_rect_x1+1，后面为什么要加1，有评论说是计算交集像素值
    # TODO 这里inter_rect_x2是相对于特征层的位置，这里不再深究，就先把问题放在这里

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)     # box1的面积
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)     # box2的面积

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)     # 计算交并比

    return iou


#非极大值抑制
def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    使用置信度过滤和非极大值抑制
    Args:
        prediction: 模型的预测结果（经过解码后的数据），
                如果要预测80个类别，那么prediction的维度为torch.Size([batch_size, num_anchors, 85])
        conf_thres: 置信度阈值
        nms_thres: NMS阈值

    Returns:一个列表，其元素个数为batch_size，每个元组都是torch张量，对应每张图片经过两轮筛选后的结果，
            如果图片中存在目标，那么对应的元素维度为(num_objs, 7)，
                    7列的内容分别为：x1, y1, x2, y2, obj_conf, class_conf, class_pred，
            其中坐标为归一化后的数值，如果图片中不存在目标，那么对应的元素为None

    """

    # 将解码结果的中心点坐标和宽高转换成左上角和右下角的坐标
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]     # len(prediction))是batch_size，即图片数量
    for image_i, image_pred in enumerate(prediction):
        """第一轮过滤"""
        # 利用目标置信度（即对应的预测框存在要检测的目标的概率）做第一轮过滤
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]

        # 如果当前图片中，所有目标的置信度都小于阈值，那么就进行下一轮循环，检测下一张图片
        if not image_pred.size(0):
            continue

        # 目标置信度乘以各个类别的概率，并对结果取最大值，获得各个预测框的score
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # image_pred[:, 4]是置信度，image_pred[:, 5:].max(1)[0]是各个类别的概率最大值

        # 将image_pred中的预测框按score从大到小排序
        image_pred = image_pred[(-score).argsort()]
        # argsort()是将(-score)中的元素从小到大排序，返回排序后索引
        # 将(-score)中的元素从小到大排序，实际上是对score从大到小排序
        # 将排序后的索引放入image_pred中作为索引，实际上是对本张图片中预测出来的目标，按score从大到小排序

        # 获得第一轮过滤后的各个预测框的类别概率最大值及其索引
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        # class_confs 类别概率最大值，class_preds 预测类别在80个类别中的索引

        # 将各个目标框的上下角点坐标、目标置信度、类别置信度、类别索引串起来
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # 经过上条命令之后，detections的维度为(number_pred, 7)
        # 7列的内容分别为：x1, y1, x2, y2, obj_conf, class_conf, class_pred

        """第二轮过滤"""
        keep_boxes = []     # 用来存储符合要求的目标框
        while detections.size(0):   # 如果detections中还有目标
            """以下标注是执行第一轮循环时的标注，后面几轮以此类推"""

            # 获得与第一个box（最大score对应的box）具有高重叠的预测框的布尔索引
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            # bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4])返回值的维度为(num_objects, )
            # bbox_iou的返回值与非极大值抑制的阈值相比较，获得布尔索引
            # 即剩下的边框中，只有detection[0]的iou大于nms_thres的，才抑制，即认为这些边框与detection[0]检测的是同一个目标

            # 获得与第一个box相同类别的预测框的索引
            label_match = detections[0, -1] == detections[:, -1]
            # 布尔索引，获得所有与detection[0]相同类别的对象的索引

            # 获得需要抑制的预测框的布尔索引
            invalid = large_overlap & label_match   # &是位运算符，两个布尔索引进行位运算
            # 经过第一轮筛选后的剩余预测框，如果同时满足和第一个box有高重叠、类别相同这两个条件，那么就该被抑制
            # 这些应该被抑制的边框，其对应的索引即为无效索引

            # 获得被抑制预测框的置信度
            weights = detections[invalid, 4:5]

            # 加权获得最后的预测框坐标
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            # 上面的命令是将当前边框，和被抑制的边框进行加权，
            # 类似于好几个边框都检测到了同一张人脸，将这几个边框的左上角点横坐标x进行加权（按照置信度加权），
            # 获得最后边框的x，对左上角点的纵坐标y，以及右下角点的横纵坐标也进行加权处理
            # 其他的obj_conf, class_conf, class_pred则使用当前box的

            keep_boxes += [detections[0]]       # 将第一个box加入到 keep_boxes 中
            detections = detections[~invalid]   # 去掉无效的预测框，更新detections
            
            
        if keep_boxes:                          # 如果keep_boxes不是空列表
            keep_boxes=avg_list(keep_boxes)
            output[image_i] = torch.stack(keep_boxes)   # 将目标堆叠，然后加入到列表
            # 假设NMS之后，第i张图中有num_obj个目标，那么torch.stack(keep_boxes)的结果是就是一个(num_obj, 7)的张量，没有图片索引

        # 如果keep_boxes为空列表，那么output[image_i]则未被赋值，保留原来的值（原来的为None）
    return output

def avg_list ( box):
    z=[]
    for  col in zip(*box):    # 二维列表取列
        z.append(sum(col)/len(box))
    
    return z

  
def draw_boxes2(image, outputs, font_file, class_names):
    """
    在图片上画框
    Args:
        image: 要画框的图片，PIL.Image.open的返回值
        outputs: 一个列表，NMS后的结果，其中的坐标为归一化后的坐标
        font_file:字体文件路径
        class_names:类名列表
        colors_list:颜色列表

    Returns:

    """
    # 根据图片的宽，动态调整字体大小
    font_size = np.floor(3e-2 * image.size[1] + 0.5).astype('int32')
    font = ImageFont.truetype(font=font_file, size=font_size)  # 创建字体对象，包括字体和字号
    draw = ImageDraw.Draw(image) 
    obj=outputs[0]# 将letter
    box=obj[: 4]*640
    cls_index = int(obj[6]) 
    score = obj[4] * obj[5]         # score，可以理解为类别置信度
    x1, y1, x2, y2 = map(int, box)          # 转化为整数
    pred_class = class_names[cls_index]     # 目标类别名称
    color = 'red'                           # TODO 具体使用时，还得改成colors_list[cls_index]

    """组建要显示的文字信息"""
    label = ' {} {:.2f}'.format(pred_class, score)
    print(label, x1, y1, x2, y2)

    """获得文字的尺寸"""
    label_size = draw.textsize(label, font)
    label = label.encode('utf-8')

    """防止文字背景框在上边缘越界"""
    if y1 - label_size[1] >= 0:
        text_origin = np.array([x1, y1 - label_size[1]])
    else:
        # 如果越界，则将文字信息写在边框内部
        text_origin = np.array([x1, y1 + 1])

    """绘制边框"""
    thickness = 2               # 边框厚度
    for i in range(thickness):  # 根据厚度确定循环的执行次数
        draw.rectangle([x1 + i, y1 + i, x2 - i, y2 - i], outline=color)  # colors[cls_index]

    """绘制文字框"""
    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color)   # 背景
    draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)              # 文字
    del draw   
    return image
    

 
def draw_boxes(image, outputs, font_file, class_names):
    """
    在图片上画框
    Args:
        image: 要画框的图片，PIL.Image.open的返回值
        outputs: 一个列表，NMS后的结果，其中的坐标为归一化后的坐标
        font_file:字体文件路径
        class_names:类名列表
        colors_list:颜色列表

    Returns:

    """
    # 根据图片的宽，动态调整字体大小
    font_size = np.floor(3e-2 * image.size[1] + 0.5).astype('int32')
    font = ImageFont.truetype(font=font_file, size=font_size)  # 创建字体对象，包括字体和字号
    draw = ImageDraw.Draw(image)                # 将letterbox_img作为画布
    
    for output in outputs:                      # ouput是每张图片的检测结果，当然这里batch_size为1就是了
        if output is not None:
            for obj in output:                  # 一张图片可能有多个目标，obj就是其中之一
                """从obj中获得信息"""
                box = obj[:4] * 640             # 将归一化后的坐标转化为输入图片（letterbox_img）中的坐标
                cls_index = int(obj[6])         # 类别索引
                score = obj[4] * obj[5]         # score，可以理解为类别置信度
                x1, y1, x2, y2 = map(int, box)          # 转化为整数
                pred_class = class_names[cls_index]     # 目标类别名称
                color = 'red'                           # TODO 具体使用时，还得改成colors_list[cls_index]

                """组建要显示的文字信息"""
                label = ' {} {:.2f}'.format(pred_class, score)
                print(label, x1, y1, x2, y2)

                """获得文字的尺寸"""
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')

                """防止文字背景框在上边缘越界"""
                if y1 - label_size[1] >= 0:
                    text_origin = np.array([x1, y1 - label_size[1]])
                else:
                    # 如果越界，则将文字信息写在边框内部
                    text_origin = np.array([x1, y1 + 1])

                """绘制边框"""
                thickness = 2               # 边框厚度
                for i in range(thickness):  # 根据厚度确定循环的执行次数
                    draw.rectangle([x1 + i, y1 + i, x2 - i, y2 - i], outline=color)  # colors[cls_index]

                """绘制文字框"""
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color)   # 背景
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)              # 文字
            del draw

    return image

def resize_image(image, size, letterbox_image):
    """
        对输入图像进行resize
    Args:
        image:PIL.Image.open的返回值，RGB三通道图像
        size:目标尺寸
        letterbox_image: bool 是否进行letterbox变换

    Returns:指定尺寸的图像

    """
    from PIL import Image
    image=Image.open(image)
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w/iw, h/ih)     # 获得长边对应的高宽比
        nw = int(iw*scale)          # 新的宽
        nh = int(ih*scale)          # 新的高

        image = image.resize((nw, nh), Image.BICUBIC)           # 调整原图的大小
        new_image = Image.new('RGB', size, (128, 128, 128))     # 生成画布
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))          # 将调整后的图像放入画布中
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image





if __name__ == '__main__':
    """模型的导入"""
    model=YOLOX_ONNX(onnx_path="./yolox.onnx")
    img_path='./data_image/test.jpg'
    font_path='./simhei.ttf'
    class_path='./classes.txt'
    '''执行前向操作预测输出'''
    # 超参数设置
    img_size=(640,  640) #图片缩放大小
    conf_thres=0.5 #置信度阈值
    iou_thres=0.3 #iou阈值
    class_num=1 #类别数
    letterbox_img=resize_image(img_path, img_size, True)
    stride=[8, 16, 32]

    anchor_list= [[10,13, 16,30, 33,23],[30,61, 62,45, 59,119], [116,90, 156,198, 373,326]]
    anchor = np.array(anchor_list).astype(np.float64).reshape(3,-1,2)

    area = img_size[0] * img_size[1]
    size = [int(area / stride[0] ** 2), int(area / stride[1] ** 2), int(area / stride[2] ** 2)]
    feature = [[int(j / stride[i]) for j in img_size] for i in range(3)]
    
    with open(class_path, 'r') as f:
        class_names = f.read().split("\n")

    # 读取图片
    src_img=cv2.imread(img_path)
    src_size=src_img.shape[:2]

    # 图片填充并归一化
    img=model.letterbox(src_img, img_size, stride=32)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)


    # 归一化
    img=img.astype(dtype=np.float32)
    img/=255.0

    # # BGR to RGB
    # img = img[:, :, ::-1].transpose(2, 0, 1)
    # img = np.ascontiguousarray(img)

    # 维度扩张
    img=np.expand_dims(img,axis=0)

    # 前向推理
    start=time.time()
    input_feed=model.get_input_feed(img)
    pred=model.onnx_session.run(output_names=model.output_name, input_feed=input_feed)
    #output=model.infer(img_path="./img/1.jpg")
    # # 模型路径
    # imgdata = image_process('./img/1.jpg')
    
    # sess = rt.InferenceSession('yolox.onnx')
    # input_name = sess.get_inputs()[0].name  
    # output_name = sess.get_outputs()[0].name
    
    # input_shape = (640, 640)
    
    
    output=decode_outputs(pred, img_size)
    output=non_max_suppression(output)
    letterbox_img = draw_boxes2(letterbox_img, output,  font_path, class_names=class_names)    # 画框
    letterbox_img.show()
    