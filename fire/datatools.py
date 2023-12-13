 
from PIL import Image
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
import cv2
import albumentations as A
import json
import platform


from fire.dataaug_user import TrainDataAug, TestDataAug





##### Common
def getFileNames(file_dir, tail_list=['.png','.jpg','.JPG','.PNG']): 
        L=[] 
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if os.path.splitext(file)[1] in tail_list:
                    L.append(os.path.join(root, file))
        return L


def img_add(img_src, img_main, mask_src):
    if len(img_main.shape) == 3:
        h, w, c = img_main.shape
    elif len(img_main.shape) == 2:
        h, w = img_main.shape
    mask = np.asarray(mask_src, dtype=np.uint8)
    sub_img01 = cv2.add(img_src, np.zeros(np.shape(img_src), dtype=np.uint8), mask=mask)

    mask_02 = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_02 = np.asarray(mask_02, dtype=np.uint8)
    sub_img02 = cv2.add(img_main, np.zeros(np.shape(img_main), dtype=np.uint8),
                        mask=mask_02)

    img_main = img_main - sub_img02 + cv2.resize(sub_img01, (img_main.shape[1], img_main.shape[0]),interpolation=cv2.INTER_NEAREST)

    return img_main



######## dataloader

class TensorDatasetTrainClassify(Dataset):
    def __init__(self, data, cfg, mode, aug=None,transform=None):
        self.data = data
        self.cfg = cfg
        self.aug = aug
        self.mode = mode
        self.transform = transform

    def read_data(self, index):
        img = cv2.imread(self.data[index])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.cfg['img_size'])
        # img1 = A.Sharpen(alpha=1.0, lightness=1.0, always_apply=True)(image=img1)['image']

        label = cv2.imread(self.data[index].replace("train_pic","train_tag")[:-3]+'png', cv2.IMREAD_GRAYSCALE)#/255.
        # unique_values, indices_list = np.unique(label, return_index=True)
        # print(unique_values)
        label = cv2.resize(label, self.cfg['img_size'], interpolation=cv2.INTER_NEAREST)
        # unique_values, indices_list = np.unique(label, return_index=True)
        # print(unique_values)
        label[label==5] = 0
        label[label==6] = 5
        label[label==7] = 0
        label[label==8] = 6
        label[label==9] = 0
        label[label==10] = 0
        label[label==11] = 0
        label[label==12] = 0
        label[label==13] = 7
        label[label==14] = 0
        label[label==15] = 0
        label[label==16] = 0
        label[label==17] = 0
        label[label==18] = 0
        label[label==19] = 0
        # if self.mode=='train':
        #     label2 = cv2.imread(self.data[index].replace("Image1","label2")[:-3]+"png", cv2.IMREAD_GRAYSCALE)
        #     label2 = cv2.resize(label2, self.cfg['img_size'])
        #     label = label*0.5+label2*0.5

        return img,label


    def copy_paste(self, img1, img2, label, p=1):
        if random.random() < 1-p:
           return img1, img2, label
        roi_file = random.choice(self.data)
        #print(roi_file)

        roi_img1 = cv2.imread(roi_file)
        roi_img1 = cv2.resize(roi_img1, self.cfg['img_size'])

        roi_img2 = cv2.imread(roi_file.replace("A","B"))
        roi_img2 = cv2.resize(roi_img2, self.cfg['img_size'])

        roi_label = cv2.imread(roi_file.replace("A","label")[:-3]+"png", cv2.IMREAD_GRAYSCALE)
        roi_label = cv2.resize(roi_label, self.cfg['img_size'])

        img1 = cv2.resize(img1, self.cfg['img_size'])
        img2 = cv2.resize(img2, self.cfg['img_size'])
        label = cv2.resize(label, self.cfg['img_size'])

        #print(label.shape,roi_label.shape)
        '''and_label = cv2.bitwise_and(roi_label,label)
        num = cv2.countNonZero(and_label)
        if num > 0 : return img1, img2, label'''

        new_label = cv2.bitwise_or(roi_label,label)

        new_img1 = img_add(roi_img1, img1, roi_label)
        new_img2 = img_add(roi_img2, img2, roi_label)

        return new_img1, new_img2, new_label


    def __getitem__(self, index):

        img,label = self.read_data(index)

        # if self.mode == 'train':
        #     img1,img2,label = self.copy_paste(img1, img2, label)
        
        if self.aug is not None:
            img,label = self.aug(img,label)


        #mixup  cutmix  mosaic
        #mixup  cutmix  mosaic
        # rd = random.random()
        # if self.mode=='train' and rd<0.5:
        # #     #mosaic
        #     h,w = self.cfg['img_size']
        #     add_indexes = random.choices([x for x in range(len(self.data)-1)], k=3)

        #     img1_2,img2_2,label_2 = self.read_data(add_indexes[0])
        #     img1_2,img2_2,label_2 = self.copy_paste(img1_2, img2_2, label_2)
        #     if self.aug is not None:
        #         img1_2,img2_2,label_2 = self.aug(img1_2,img2_2,label_2)

        #     img1_3,img2_3,label_3 = self.read_data(add_indexes[1])
        #     img1_3,img2_3,label_3 = self.copy_paste(img1_3, img2_3, label_3)
        #     if self.aug is not None:
        #         img1_3,img2_3,label_3 = self.aug(img1_3,img2_3,label_3)

        #     img1_4,img2_4,label_4 = self.read_data(add_indexes[2])
        #     img1_4,img2_4,label_4 = self.copy_paste(img1_4, img2_4, label_4)
        #     if self.aug is not None:
        #         img1_4,img2_4,label_4 = self.aug(img1_4,img2_4,label_4)

        #     img4_1 = np.zeros((h*2, w*2,3))
        #     img4_2 =  np.zeros((h*2, w*2,3))
        #     label4 = np.zeros((h*2, w*2))

        #     img4_1[:h,:w] = img1 
        #     img4_2[:h,:w] = img2 
        #     label4[:h,:w] = label 

        #     img4_1[:h,w:] = img1_2 
        #     img4_2[:h,w:] = img2_2 
        #     label4[:h,w:] = label_2 

        #     img4_1[h:,:w] = img1_3 
        #     img4_2[h:,:w] = img2_3 
        #     label4[h:,:w] = label_3 

        #     img4_1[h:,w:] = img1_4 
        #     img4_2[h:,w:] = img2_4 
        #     label4[h:,w:] = label_4 

        #     crop_x = random.randint(int(w*0.3),int(w*0.7))
        #     crop_y = random.randint(int(h*0.3),int(h*0.7))
        #     crop_w = random.randint(int(w-w*0.2),int(w+w*0.2))
        #     crop_h = random.randint(int(h-h*0.2),int(h+h*0.2))
        #     # print(img4_1.shape, crop_x,crop_y,crop_w,crop_h)
        #     img1 = img4_1[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        #     img2 = img4_2[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        #     label = label4[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

        #     img1 = cv2.resize(img1, self.cfg['img_size'])
        #     img2 = cv2.resize(img2, self.cfg['img_size'])
        #     label = cv2.resize(label, self.cfg['img_size'])
        #     ###mixup
        #     # mid = random.randint(0,len(self.data)-1)
        #     # img1_2 = cv2.imread(self.data[mid])
        #     # img1_2 = cv2.resize(img1_2, self.cfg['img_size'])

        #     # img2_2 = cv2.imread(self.data[mid].replace("Image1","Image2"))
        #     # img2_2 = cv2.resize(img2_2, self.cfg['img_size'])

        #     # label_2 = cv2.imread(self.data[mid].replace("Image1","label1")[:-3]+"png", cv2.IMREAD_GRAYSCALE)
        #     # label_2 = cv2.resize(label_2, self.cfg['img_size'])

        #     # if self.aug is not None:
        #     #     img1_2,img2_2,label_2 = self.aug(img1_2,img2_2,label_2)

        #     # lam = random.random()
        #     # img1 = lam*img1+(1-lam)*img1_2
        #     # img2 = lam*img2+(1-lam)*img2_2
        #     # label = lam*label+(1-lam)*label_2
        # # # elif self.mode=='train' and rd<0.8:
        #         #cutmix
        # #     lam = random.random()
        # #     hw_range = int(lam*img2_2.shape[1])
        # #     img1[hw_range:,hw_range:] = img1_2[hw_range:,hw_range:]
        # #     img2[hw_range:,hw_range:] = img2_2[hw_range:,hw_range:]
        # #     label[hw_range:,hw_range:] = label_2[hw_range:,hw_range:]


        label = label.reshape(label.shape[0],label.shape[1],1)


        #print(img)
        if self.transform is not None:
            result  = self.transform(image = img)
            img = result['image']

        # y_onehot = [0,0]
        # y_onehot[y] = 1
        #print(img,label)
        #bb
        img = img.transpose([2,0,1])
        label = label.transpose([2,0,1])
        return img, torch.from_numpy(label).long(), self.data[index]
        
    def __len__(self):
        return len(self.data)


class TensorDatasetTestClassify(Dataset):

    def __init__(self, data, cfg, aug=None, transform=None):
        self.data = data
        self.cfg = cfg
        self.aug=aug
        self.transform = transform


    def __getitem__(self, index):

        img = cv2.imread(self.data[index])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.cfg['img_size'])

        if self.transform is not None:
            result  = self.transform(image = img)
            img = result['image']
        #print(img)
        img = img.transpose([2,0,1])
        #print(img.shape,img)
        return img, self.data[index]

    def __len__(self):
        return len(self.data)


###### 3. get data loader 




def getDataLoader(mode, input_data, cfg):

    # my_normalize = getNormorlize(cfg['model_name'])
    # my_normalize = transforms.Normalize([0.485, 0.456, 0.406,0.485, 0.456, 0.406], 
    #                         [0.229, 0.224, 0.225,0.229, 0.224, 0.225])

    my_normalize = A.Normalize(mean=(0.485, 0.456, 0.406), 
                            std=(0.229, 0.224, 0.225))


    data_aug_train = TrainDataAug(cfg['img_size'])
    data_aug_test = TestDataAug(cfg['img_size'])


    if mode=="test":
        my_dataloader = TensorDatasetTestClassify

        test_loader = torch.utils.data.DataLoader(
                                my_dataloader(input_data[0],
                                    cfg,
                                    data_aug_test,
                                    A.Compose([
                                        # data_aug_test,
                                        # transforms.ToTensor(),
                                        my_normalize
                                    ])
                ), batch_size=cfg['test_batch_size'], shuffle=False, 
                num_workers=cfg['num_workers'], pin_memory=True
            )

        return test_loader


    elif mode=="trainval":
        my_dataloader = TensorDatasetTrainClassify


        train_loader = torch.utils.data.DataLoader(
                                        my_dataloader(input_data[0],
                                            cfg,
                                            'train',
                                            data_aug_train,
                                            A.Compose([
                                                # data_aug_train,
                                                # transforms.ToTensor(),
                                                my_normalize,
                                        ])),
                                batch_size=cfg['batch_size'], 
                                shuffle=True, 
                                num_workers=cfg['num_workers'], 
                                drop_last=True,
                                pin_memory=True)


        val_loader = torch.utils.data.DataLoader(
                                        my_dataloader(input_data[1],
                                            cfg,
                                            'val',
                                            data_aug_test,
                                            A.Compose([
                                                # data_aug_test,
                                                # transforms.ToTensor(),
                                                my_normalize
                                        ])),
                                batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=True)
        return train_loader, val_loader


