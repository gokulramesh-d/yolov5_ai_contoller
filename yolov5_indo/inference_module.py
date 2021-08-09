import sys
sys.path.append("D:\deployment")

import os
import glob
import cv2
from detect_module import *
# from classifier_module import *
from opt_module import *
from common_utils import *
import torch
from datetime import  datetime
import uuid




@singleton
class Inference:
    def __init__(self):
        self.opt = opt_config()
        self.detector = load_detector(self.opt)
        # self.classifier = load_classifier(self.opt)
        self.input_frame = None
        self.predicted_frame = None
        self.detector_predictions = None
        self.classifier_predictions = None
        self.device = self.opt.device
        self.crop = None
    def predict(self):
        # self.crop = detector_crop(self.opt,self.detector,self.input_frame)
        # cv2.imwrite("./crops/"+str(uuid.uuid1())+".jpg",self.crop)
        # print("inside prediction")
        self.predicted_frame, self.detector_predictions = detector_inference(self.opt,model = self.detector ,frame = self.input_frame )
        
        # self.classifier_predictions = get_classifier_prediction(self.opt,frame = self.input_frame,learn = self.classifier ,device = self.device)
        self.predicted_frame = cv2.resize(self.predicted_frame,(640,480))
        return self.predicted_frame, self.detector_predictions #, self.classifier_predictions

    
def test(in_dir,out_dir):
    with torch.no_grad():
        predictor = Inference()
        for im in glob.glob(os.path.join(in_dir,"*.jpg")):
            
            img = cv2.imread(im)
            
            # print(f'Image shape {img.shape}')
            file_name = im.split("/")[-1]
            t0 = datetime.now()
            predictor.input_frame  = img
            predicted_frame, detector_predictions  = predictor.predict()
            t1 = datetime.now()
            print(f'Time taken for prediction of one frame {(t1-t0).total_seconds()} sec')
            cv2.imwrite(os.path.join(out_dir,file_name),predicted_frame)
            
            
            # print(im ,detector_predictions , classifier_predictions)



if __name__ == '__main__' :
    test("./data/images","./data/out")





