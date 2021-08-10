import sys
sys.path.append("D:\deployment")

import os
import glob
import cv2
from detect_module import *
# from classifier_module import *
from config_module import *
from common_utils import *
import torch
from datetime import  datetime
import uuid




@singleton
class Inference:
    def __init__(self):
        self.opt = opt_config()
        self.device = device = select_device(self.opt.device)
        self.half = self.opt.half
        self.crop = self.opt.crop
        self.cropped_frame = None
        self.input_frame = None
        self.predicted_frame = None
        self.detector_predictions = None
        self.detector , self.detector_stride , self.detector_names  = load_detector(path = self.opt.detector_weights_path,
                                                        half = self.half,device = self.device ,
                                                        imgsz = self.opt.detector_input_image_size)

        if self.crop:
            if self.opt.separate_crop_model:
                self.cropper,self.cropper_stride ,self.cropper_names = load_detector(path = self.opt.crop_detector_weights_path,
                                                                    half = self.half,device = self.device ,
                                                                    imgsz = self.opt.detector_input_image_size)
            else:
                self.cropper,self.cropper_stride ,self.cropper_names = self.detector , self.detector_stride , self.detector_names 
    
    def get_cropped_frame(self):
        self.cropped_frame = detector_get_crop(self.opt ,im0 = self.input_frame , class_name = self.opt.crop_class ,
                                                 names = self.cropper_names , padding = self.opt.padding ,
                                                 img_size = self.opt.detector_input_image_size ,stride = self.cropper_stride , 
                                                 model= self.cropper , device = self.device ,
                                                 half = self.half)

        print(self.cropped_frame.shape , "cropped!!")
    def detector_predict(self):
        if bool(self.crop) & bool(len(self.cropped_frame)):
            print(self.cropped_frame.shape)
            self.predicted_frame, self.detector_predictions = detector_get_inference(opt = self.opt,
                                                                                    im0 = self.cropped_frame , names = self.detector_names,
                                                                                    img_size = self.opt.detector_input_image_size ,
                                                                                    stride = self.detector_stride ,
                                                                                    model= self.detector , device = self.device,
                                                                                    half = self.half)
        else:
            self.predicted_frame, self.detector_predictions = detector_get_inference(opt = self.opt,
                                                                                    im0 = self.input_frame , names = self.detector_names,
                                                                                    img_size = self.opt.detector_input_image_size ,
                                                                                    stride = self.detector_stride ,
                                                                                    model= self.detector , device = self.device,
                                                                                    half = self.half)
        self.predicted_frame = cv2.resize(self.predicted_frame,(640,480))
        return self.predicted_frame, self.detector_predictions #, self.classifier_predictions

    
def test(in_dir,out_dir):
    with torch.no_grad():
        predictor = Inference()
        for im in glob.glob(os.path.join(in_dir,"*.jpg")):
            img = cv2.imread(im)
            file_name = im.split("/")[-1]
            t0 = datetime.now()
            predictor.input_frame  = img
            if predictor.crop :
                predictor.get_cropped_frame()
                print(predictor.cropped_frame.shape)
                cv2.imwrite(os.path.join(out_dir,"crop"+file_name),predictor.cropped_frame)
            predicted_frame, detector_predictions  = predictor.detector_predict()
            t1 = datetime.now()
            print(f'Time taken for prediction of one frame {(t1-t0).total_seconds()} sec')
            cv2.imwrite(os.path.join(out_dir,file_name),predicted_frame)
            
            
            # print(im ,detector_predictions , classifier_predictions)



if __name__ == '__main__' :
    test("./data/images","./data/out")





