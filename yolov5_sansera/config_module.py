import os
import sys



class opt():
    def __init__(self):
        self.detector_weights_path = 'yolov5s.pt'
        self.image_size = 640
        self.common_conf_thres = 0.25 
        self.iou_thres = 0.45
        self.max_det = 1000
        self.device = ""
        self.line_thickness = 3
        self.hide_labels = False
        self.hide_conf = False
        self.half = True
        self.crop = False
        self.crop_class = None
        self.min_crop_size = None
        self.max_crop_size = None
        self.crop_conf = 0.25
        self.crop_iou = 0.25
        self.padding  = 0
        self.crop_hide_labels = True
        self.crop_hide_conf = True
        self.classes = None
        
        self.defects = []
        self.feature = []
        self.individual_thres = {}
        self.visualize = False


    # parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    # parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    # parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='show results')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    # parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--visualize', action='store_true', help='visualize features')
    # parser.add_argument('--update', action='store_true', help='update all models')
    # parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    # parser.add_argument('--name', default='exp', help='save results to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    # parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    # parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    # parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # opt = parser.parse_args()