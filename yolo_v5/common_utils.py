import os
import sys
from PIL import Image
# import imagehash
import cv2
import argparse
import shutil
# import redis
# from pymongo import MongoClient
# from bson import ObjectId
import json
from ai_settings import *
import ai_settings as settings
import datetime

def cv2_pil(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil

# def identify_similar_img(frame1, frame2):

#     # hash0 = imagehash.average_hash(Image.open('./images/batch_1_3.jpg')) 
#     # hash1 = imagehash.average_hash(Image.open('./images/batch_1_38_affine.jpg')) 
#     hash0 = imagehash.average_hash(cv2_pil(frame1))
#     hash1 = imagehash.average_hash(cv2_pil(frame2))
#     cutoff = 5
#     print(hash0 , hash1  , hash0 - hash1 )
#     if hash0 - hash1 < cutoff:
#         print('images are similar')
#         similar_images = True
#     else:
#         print('images are not similar')
#         similar_images = False
#     return similar_images



def singleton(cls):
    """
    This is a decorator which helps to create only 
    one instance of an object in the particular process.
    This helps the preserve the memory and prevent unwanted 
    creation of objects.
    """
    instances = {}
    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return getinstance


@singleton    
class RedisKeyBuilderWorkstation():
    def __init__(self):
        # self.wid = get_workstation_id('livis//workstation_settings//settings_workstation.json')
        self.workstation_name = "WS_01" #get_workstation_by_id(self.wid)
    def get_key(self, camera_id, identifier):
        return "{}_{}_{}".format(self.workstation_name, str(camera_id), identifier)


import pickle

@singleton
class CacheHelper():
    def __init__(self):
        # self.redis_cache = redis.StrictRedis(host="164.52.194.78", port="8080", db=0, socket_timeout=1)
        self.redis_cache = redis.StrictRedis(host=settings.REDIS_CLIENT_HOST, port=settings.REDIS_CLIENT_PORT, db=0, socket_timeout=1)
        settings.REDIS_CLIENT_HOST
        print("REDIS CACHE UP!")

    def get_redis_pipeline(self):
        return self.redis_cache.pipeline()
    
    #should be {'key'  : 'value'} always
    def set_json(self, dict_obj):
        try:
            k, v = list(dict_obj.items())[0]
            v = pickle.dumps(v)
            return self.redis_cache.set(k, v)
        except redis.ConnectionError:
            return None

    def get_json(self, key):
        try:
            temp = self.redis_cache.get(key)
            #print(temp)\
            if temp:
                temp= pickle.loads(temp)
            return temp
        except redis.ConnectionError:
            return None
        return None

    def execute_pipe_commands(self, commands):
        #TBD to increase efficiency can chain commands for getting cache in one go
        return None

@singleton
class MongoHelper:
    try:
        client = None
        def __init__(self):
            if not self.client:
                self.client = MongoClient(host=MONGO_SERVER_HOST, port=MONGO_SERVER_PORT)
            self.db = self.client[MONGO_DB]

        def getDatabase(self):
            return self.db

        def getCollection(self, cname, create=False, codec_options=None):
            _DB = MONGO_DB
            DB = self.client[_DB]
            if cname in MONGO_COLLECTIONS:
                if codec_options:
                    return DB.get_collection(MONGO_COLLECTIONS[cname], codec_options=codec_options)
                return DB[MONGO_COLLECTIONS[cname]]
            else:
                return DB[cname]
    except:
        pass            
# @singleton
# class MongoHelper_1:
#     def _init_(self):
#         self.myclient = None
#         # if not self.client:
#         self.mydb = None

#     def get_db(self):
#         self.myclient = MongoClient(f"mongodb://{MONGO_SERVER_HOST}:{MONGO_SERVER_PORT}/")
#         self.mydb = self.myclient[str(MONGO_DB)]
#         # pass
#         return self.mydb
#     def get_collection(self,collection_name):
#         # myclient = pymongo.MongoClient(f"mongodb://{MONGO_SERVER_HOST}:{MONGO_SERVER_PORT}/")
#         self.mydb = self.get_db()
#         self.mycol = self.mydb[str(collection_name)]
#         return self.mycol
#     def insert_one_col(self,collection_name , mydict):
#         self.mycol = self.get_collection(collection_name)
#         _id = self.mycol.insert_one(mydict)
#         return str(_id)

def insert_inspection_col(collection_name ,mydict ):
    myclient = MongoClient("mongodb://localhost:27017/")
    mydb = myclient["Indo_trial"]
    mycol = mydb[str(collection_name)]
    # mydict = { "name": "John", "address": "Highway 37" }
    print(mycol ,mydict )
    _x = mycol.insert_one(mydict)
    return _x




def create_temp_folder1(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)  # delete existing lotmark folder
    os.makedirs(directory)        # creating new lotmark folder

def create_folder(directory):
    if os.path.exists(directory):
        pass
    else:
        os.makedirs(directory) 
    print(f'{directory} is created!!')


#Key Builder
def read_json_file(json_file):
    with open(json_file,'r') as f:
        data = json.load(f)
        f.close()
        return data

def get_workstation_id(json_file):
    workstation_dict = read_json_file(json_file)
    workstation_id = workstation_dict['wid']
    return workstation_id

def get_indo_running_process():
    mp = MongoHelper().getCollection('inspection_summary')
    run_process = mp.find_one({"status" : "started"})

    resp = run_process["_id"]
    if resp :
        return resp
    else :
        {}

def store_input_image(frame , status ,opt , img_name ,stage):
    x = datetime.datetime.now()
    date = x.strftime("%d_%m_%Y")
    root_dir = os.path.join(opt.input_image_path ,str(stage))
    storage_dir = os.path.join(root_dir ,date)
    if status :
        folder = os.path.join(storage_dir  , "accepted")
    else:
        folder = os.path.join(storage_dir , "rejected")
    create_folder(directory = folder)
    img_path = folder + "/"+ str(img_name)+".jpg"

    cv2.imwrite(img_path , frame) 
    return img_path
        
def store_output_image(frame , status ,opt , img_name ,stage):
    x = datetime.datetime.now()
    date = x.strftime("%d_%m_%Y")
    # storage_dir = os.path.join(opt.output_image_path,date)
    root_dir = os.path.join(opt.output_image_path ,str(stage))
    storage_dir = os.path.join(root_dir ,date)
    if status :
        folder = os.path.join(storage_dir , "accepted")
    else:
        folder = os.path.join(storage_dir  , "rejected")
    create_folder(directory = folder)
    img_path = folder + "/"+ str(img_name)+".jpg"
    cv2.imwrite(img_path , frame) 
    return img_path

# data = {
#         "active": True,
#         "_id": "5ebb930e4c2ee3532862454b",
#         "workstation_name": "WS_01",
#         "workstation_ip":settings.REDIS_CLIENT_HOST,#"192.168.0.2"
#         "client_port" : settings.REDIS_CLIENT_PORT,#"6379"
#         "camera_config": {
#             "cameras": [
#                 {
#                     "camera_name": "left_camera",
#                 "camera_id": 3,
#                     "focus": 50#15
#                 },
#                  {
#                      "camera_name": "right_camera",
#                      "camera_id": 0,
#                      "focus": 50
#                  },
#                   {
#                       "camera_name": "mid_camera",
#                       "camera_id": 1,
#                       "focus": 10
#                   },
#                   {
#                       "camera_name": "seg_camera",
#                       "camera_id": 2,
#                       "focus": 20
#                   }
                
#                 ]}
# }
    
