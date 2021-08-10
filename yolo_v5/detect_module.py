import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages , image_preprocess
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box , get_crop
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync




def load_detector(path,half,device,imgsz):
    device = select_device(device)
    half &= device.type != 'cpu' 
    model = attempt_load(weights = path, map_location=device)
    stride = int(model.stride.max())
    names = model.module.names if hasattr(model, 'module') else model.names
    #names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Run inference
    if True and device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    print("Model loaded!!!")
    return model , stride , names

def detector_get_crop(opt ,im0, class_name , names, padding ,img_size ,stride , model, device ,half):
    # img0 = cv2.imread(path)
    print("Inside get crop")
    crop = []
    img = image_preprocess(im0 , img_size = img_size ,stride = stride)
    # img = torch.from_numpy(img).to(device)
    if True: # pt
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float() 
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]

    #Inference
    pred = model(img, augment=False, visualize=False)[0]

    # NMS
    pred = non_max_suppression(pred, conf_thres = opt.crop_conf, iou_thres = opt.crop_iou, classes = None, agnostic = False, max_det=1000)

    # Process predictions
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy()
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = None if opt.crop_hide_labels else (names[c] if opt.crop_hide_conf else f'{names[c]} {conf:.2f}')
                # plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness= opt.line_thickness)
                if names[c] == opt.crop_class :
                    crop = get_crop(xyxy, imc, gain=1.02, pad=10, square=False, BGR=True, save=True)

                # save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
    torch.cuda.empty_cache()
    if bool(opt.crop_resize):
        crop = cv2.resize(crop,(640,640))
    return crop

def detector_get_inference(opt ,im0, names,img_size  ,stride, model, device ,half ):
    # img0 = cv2.imread(path)
    print("inside inference!!")
    predictions = []
    img = image_preprocess(im0 , img_size = img_size ,stride = stride)
    # img = torch.from_numpy(img).to(device)
    if True: # pt
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float() 
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]

    #Inference
    pred = model(img, augment=False, visualize=False)[0]

    # NMS
    pred = non_max_suppression(pred, conf_thres = opt.crop_conf, iou_thres = opt.crop_iou, classes = None, agnostic = False, max_det=1000)

    # Process predictions
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy()
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                if names[c] in list(opt.individual_thres.keys()):
                    if conf > opt.individual_thres[names[c]] :
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        if bool(opt.defects) :
                            if names[c] in opt.defects:
                                bndbox_color = [60,20,250]#RED
                            else:
                                bndbox_color = [0,128,0]#GREEN
                        else:
                            bndbox_color = colors(c, True)
                        plot_one_box(xyxy, im0, label=label, color=bndbox_color, line_thickness= opt.line_thickness)
                        predictions.append(names[c])
                else:
                    label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                    if bool(opt.defects) :
                        if names[c] in opt.defects:
                            bndbox_color = [60,20,250]#RED
                        else:
                            bndbox_color = [0,128,0]#GREEN
                    else:
                        bndbox_color = colors(c, True)
                    plot_one_box(xyxy, im0, label=label, color=bndbox_color, line_thickness= opt.line_thickness)
                    predictions.append(names[c])
                

                # save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
    torch.cuda.empty_cache()
    return im0 , predictions