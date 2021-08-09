import argparse
import time
from pathlib import Path
import torch
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import preprocess


defects = ["short_fill","damage","crack","flow_line"]
threshold = {
            "mdv001" : 0.1 ,
            "flow_line" : 0.3 ,
            "crack" :0.1,
            "wing_distortion": 0.25,
            "short_fill" : 0.1 ,
            "proper_fill":0.35 ,
            "damage": 0.3 ,
            "blister":0.1,
            "pip_broken":0.1,
            "flash":0.2}


def load_detector(opt):
    weights  =  opt.weights
    print(weights)
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    print(device)
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model 
    if half:
        model.half()  # to FP16   

    print("model loaded!!!!!!!!!!!!!!11>>>>>>>>>>>>>>>")
    return model


# def preprocess(frame , opt):
#     img_size= opt.img_size
#     imgs = [frame]
#     s = np.stack([letterbox(x, new_shape= img_size)[0].shape for x in imgs], 0)  # inference shapes
#     rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
#     img0 = imgs.copy()
#     # Letterbox
#     img = [letterbox(x, new_shape=img_size, auto=rect)[0] for x in img0]
#     # Stack
#     img = np.stack(img, 0)
#     # Convert
#     img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
#     img = np.ascontiguousarray(img)
#     return img, img0


def detector_crop_v3(opt,model ,frame):
    # print(f'Inside inference!!!!!!!!!!!!!!!!!!!!!')
    imgsz =  opt.img_size
    # Initialize
    # set_logging()
    # print(frame.shape)
    
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    imgsz = check_img_size(imgsz, s = model.stride.max())  # check img_size


    # Set Dataloader
    cudnn.benchmark = True  # set True to speed up constant image size inference
    # dataset = LoadStreams(source= 0, img_size=imgsz)
    # dataset = preprocess(frame , opt)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    img, im0s  = preprocess(frame , opt)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    predictions = []
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if True:  # batch_size >= 1
            # im_original =  im0s[i].copy()
            im0 =  im0s[i].copy()

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                # s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = '%s %.2f' % (names[int(cls)], conf)
                # label = '%s' % (names[int(cls)])
                # if label in defects:

                if names[int(cls)] == "bearing":
                    # im0 = im0[]
                    print(xyxy)
                    print("Inside Crop!!!!!!!!!!!!!!!!!!",im0.shape)
                    im0 = get_crop(xyxy,im0)
                    print("Inside Crop!!!!!!!!!!!!!!!!!!",im0.shape)
                # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                # predictions.append(label)
    return im0 #, predictions#,im_original


    
def detector_inference_v3(opt,model ,frame):
    imgsz =  opt.img_size
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    imgsz = check_img_size(imgsz, s = model.stride.max())  # check img_size


    # Set Dataloader
    cudnn.benchmark = True  # set True to speed up constant image size inference
    # dataset = LoadStreams(source= 0, img_size=imgsz)
    # dataset = preprocess(frame , opt)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    img, im0s  = preprocess(frame , opt)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    predictions = []
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if True:  # batch_size >= 1
            # im_original =  im0s[i].copy()
            im0 =  im0s[i].copy()

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                # s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                label_conf = '%s %.2f' % (names[int(cls)], conf)
                label = '%s' % (names[int(cls)])
                if label in defects:
                    bndbox_color = [60,20,250]#RED
                else:
                    bndbox_color = [0,128,0]#GREEN
                if threshold[label] < float(conf):
                    plot_one_box(xyxy, im0, label=label, color=bndbox_color, line_thickness=3)#colors[int(cls)]
                    predictions.append(label)
                    print(label)
    return im0 , predictions#,im_original



def detector_inference(opt,model ,frame):
    imgsz =  opt.img_size
    # Initialize
    # set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Set Dataloader
    cudnn.benchmark = True  # set True to speed up constant image size inference

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    print("Before preprocess>>>>>>>>>>>>>>>")
    img, im0s = preprocess(frame , opt ,stride = stride)
    print("After preprocess>>>>>>>>>>>>>>>")
    img = torch.from_numpy(img).to(device)
    print(img.shape ,"after torch numpy")
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    print(img.shape , "after unsqueeze")
    # Inference
    # t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    # t2 = time_synchronized()

    predictions = []
    # Process detections
    for i, det in enumerate(pred):
        im0 = im0s.copy()# im0s[i].copy()
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0s[i].copy()#im0.copy() #if opt.save_crop else im0  # for opt.save_crop
        print(im0.shape)
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


        # Write results
        for *xyxy, conf, cls in reversed(det):
            # if save_txt:  # Write to file
            #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            if True:  # Add bbox to image
                c = int(cls)  # integer class
                # label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                label_conf = '%s %.2f' % (names[int(cls)], conf)
                label = '%s' % (names[int(cls)])
                if label in defects:
                    bndbox_color = [60,20,250]#RED
                else:
                    bndbox_color = [0,128,0]#GREEN]
                if threshold[label] < float(conf):
                    plot_one_box(xyxy, im0, label=label, color=bndbox_color, line_thickness=3)
                    predictions.append(label)
    torch.cuda.empty_cache()
    return im0 , predictions














from opt_module import *
import cv2
def test():

    opt = opt_config()
    frame = cv2.imread("in/a.jpg")
    model = load_detector(opt)


    with torch.no_grad():
        # if opt.update:  # update all models (to fix SourceChangeWarning)
        #     for opt.weights in ['yolov3.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt']:
        #         detect()
        #         strip_optimizer(opt.weights)
        # else:

        im0 , predictions = detect(opt = opt,model = model ,frame = frame)
        print(predictions)
        # detect()


import glob


if __name__ == '__main__':
    print("started!!!!!!!!!!!")
    from opt_module import *
    import cv2
    opt = opt_config()
    print(opt)
    # frame = cv2.imread("in/a.jpg")
    # print(frame.shape)
    model = load_detector(opt)


    with torch.no_grad():
        # if opt.update:  # update all models (to fix SourceChangeWarning)
        #     for opt.weights in ['yolov3.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt']:
        #         detect()
        #         strip_optimizer(opt.weights)
        # else:
        for im in glob.glob("./in/*.jpg"):
            frame = cv2.imread(im)
            im0 , predictions = detect(opt = opt,model = model ,frame = frame)
            print(predictions)
            im_name = f'./out/{im.split("/")[-1]}'
            print(im_name)
            cv2.imwrite(im_name,im0)
        # detect()
