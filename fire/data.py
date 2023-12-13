
import os
import random
import numpy as np
from sklearn.model_selection import KFold

import cv2
from torchvision import transforms

from fire.datatools import getDataLoader, getFileNames
from fire.dataaug_user import TrainDataAug



class FireData():
    def __init__(self, cfg):
        
        self.cfg = cfg


    def getTrainValDataloader(self):

        img_dir = "train_pic"
        all_data = getFileNames(os.path.join(self.cfg['train_path'],img_dir),['.PNG','.jpg'])

        print("[INFO] val_path is none, use kflod to split data: k=%d val_fold=%d" % (self.cfg['k_flod'],self.cfg['val_fold']))
        print("[INFO] Total images: ", len(all_data))

        all_data.sort()
        random.shuffle(all_data)

        if self.cfg['try_to_train_items'] > 0:
            all_data = all_data[:self.cfg['try_to_train_items']]

        fold_count = int(len(all_data)/self.cfg['k_flod'])
 
        if self.cfg['val_fold']==self.cfg['k_flod']:
            train_data = all_data
            val_data = all_data[:100]
        else:
            val_data = all_data[fold_count*self.cfg['val_fold']:fold_count*(self.cfg['val_fold']+1)]
            train_data = all_data[:fold_count*self.cfg['val_fold']]+all_data[fold_count*(self.cfg['val_fold']+1):]


        print(f"Final train: {len(train_data)}, val: {len(val_data)}")
        input_data = [train_data, val_data]


        ### show count
        # class_count = [0 for _ in range(20)]
        # for i in range(len(train_data)):
        #     label = cv2.imread(train_data[i].replace("train_pic","train_tag")[:-3]+'png', cv2.IMREAD_GRAYSCALE)
        #     uniques = np.unique(label).tolist()
        #     for num in uniques:
        #         class_count[num] += 1
        # print("train count:",class_count)
        # class_count = [0 for _ in range(20)]
        # for i in range(len(val_data)):
        #     label = cv2.imread(val_data[i].replace("train_pic","train_tag")[:-3]+'png', cv2.IMREAD_GRAYSCALE)
        #     uniques = np.unique(label).tolist()
        #     for num in uniques:
        #         class_count[num] += 1
        # print("val count:",class_count)

        train_loader, val_loader = getDataLoader("trainval", 
                                                input_data,
                                                self.cfg)
        return train_loader, val_loader


    def getTestDataloader(self):
        data_names = getFileNames(os.path.join(self.cfg['test_path']),['.PNG','.png','.jpg'])
        
        input_data = [data_names]
        data_loader = getDataLoader("test", 
                                    input_data,
                                    self.cfg)
        return data_loader


    def showTrainData(self, show_num = 200):
        #show train data finally to exam

        show_dir = "show_img"
        show_path = os.path.join(self.cfg['save_dir'], show_dir)
        print("[INFO] Showing traing data in ",show_path)
        if not os.path.exists(show_path):
            os.makedirs(show_path)


        img_path_list = getFileNames(self.cfg['train_path'])[:show_num]
        transform = transforms.Compose([TrainDataAug(self.cfg['img_size'])])


        for i,img_path in enumerate(img_path_list):
            #print(i)
            img = cv2.imread(img_path)
            img = transform(img)
            img.save(os.path.join(show_path,os.path.basename(img_path)), quality=100)

    