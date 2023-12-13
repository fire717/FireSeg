import time
import gc
import os
import datetime
import torch
import torch.nn as nn
import numpy as np
import cv2

# from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# from fire.pycocotools.coco import COCO
# from fire.pycocotools.cocoeval import COCOeval
# from fire.pycocotools import mask as coco_mask
import json

from fire.runnertools import getSchedu, getOptimizer, getLossFunc
from fire.runnertools import clipGradient
from fire.metrics import getF1
from fire.scheduler import GradualWarmupScheduler
from fire.utils import printDash


import numpy as np





class FireRunner():
    def __init__(self, cfg, model):

        self.cfg = cfg

        



        if self.cfg['GPU_ID'] != '' :
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)


        self.scaler = torch.cuda.amp.GradScaler()
        ############################################################
        

        # loss
        self.loss_func = getLossFunc(self.device, cfg)
        

        
        # optimizer
        self.optimizer = getOptimizer(self.cfg['optimizer'], 
                                    self.model, 
                                    self.cfg['learning_rate'], 
                                    self.cfg['weight_decay'])


        # scheduler
        self.scheduler = getSchedu(self.cfg['scheduler'], self.optimizer)
        
        if self.cfg['warmup_epoch']:
            self.scheduler = GradualWarmupScheduler(optimizer, 
                                                multiplier=1, 
                                                total_epoch=self.cfg['warmup_epoch'], 
                                                after_scheduler=self.scheduler)

        # self.img_name_weights = {}
        # self.coco_gt = None



    def freezeBeforeLinear(self, epoch, freeze_epochs = 2):
        if epoch<freeze_epochs:
            for child in list(self.model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False
        elif epoch==freeze_epochs:
            for child in list(self.model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = True
        #b


    def train(self, train_loader, val_loader):


        self.onTrainStart()

        for epoch in range(self.cfg['epochs']):

            self.freezeBeforeLinear(epoch, self.cfg['freeze_nonlinear_epoch'])

            self.onTrainStep(train_loader, epoch)

            #self.onTrainEpochEnd()

            self.onValidation(val_loader, epoch)



            if self.earlystop:
                break
        

        self.onTrainEnd()


    def predictShow(self, data_loader, save_dir):
        self.model.eval()
        correct = 0

        res_dict = {}
        with torch.no_grad():
            # pres = []
            # labels = []
            for (data, img_names) in data_loader:
                data = data.to(self.device)

                output = self.model(data)

                pred_score = nn.Sigmoid()(output[0]).data.cpu().numpy()#.reshape(1,-1)
                th = self.cfg['threshold']
                pred_score[pred_score>=th] = 1
                pred_score[pred_score<th] = 0
                pred = np.array(pred_score, dtype=np.float32)

                for i in range(len(pred)):
                    img_name = os.path.basename(img_names[i])[:-3]+'png'
                    label = cv2.imread(os.path.join("../data/train_with_seg/train/label_show", img_name), cv2.IMREAD_GRAYSCALE)

                    save_path = os.path.join(save_dir, img_name)
                    mask = pred[i][0]
                    #mask = mask[:,:,np.newaxis]
                    pre = np.array(cv2.resize(mask, (512,512))*255,dtype=np.uint8)
                    #print(label.shape,pre.shape,(label*0).shape)
                    show_img = cv2.merge([label*0,label,pre])
                    cv2.imwrite(save_path, show_img)


    def predict(self, data_loader):
        self.model.eval()
        correct = 0

        res_dict = {}
        with torch.no_grad():
            # pres = []
            # labels = []
            for (data, img_names) in data_loader:
                data = data.to(self.device)
                # print(data)
                output = self.model(data)[0]
                # print(output)
                # bb
                pred = output.data.cpu().numpy()#.reshape(1,-1)
                #pred = pred[0]
                #print(pred.shape)
                
                # pred = np.array(pred_score, dtype=np.float32)
                #print(pred)
                #pred = pred*255
                # print(pred)
                # pred = pred.astype(np.uint8)
                # print(pred)
                for i in range(len(img_names)):
                    save_path = os.path.join("output/", os.path.basename(img_names[i])[:-4]+'_pre.png')
                    mask = pred[i]
                    mask = np.argmax(mask, axis=0)

                    #show = np.argmax(mask, axis=0)
                    cv2.imwrite("pre_pre.png", np.reshape(mask, (512,512)))
                    cv2.imwrite("pre_pre_show.png", np.reshape(mask, (512,512))*20)

                    print(np.max(mask))
                    #print(mask.shape)
                    #mask = mask[:,:,np.newaxis]
                    cv2.imwrite(save_path, mask)
                    #cv2.imwrite(save_path, cv2.resize(mask, (512,512)))
                    #cv2.imwrite(save_path+"255.png", cv2.resize(mask, (512,512))*255)

    def predictRaw(self, data_loader):
        self.model.eval()
        correct = 0

        res_dict = {}
        with torch.no_grad():
            # pres = []
            # labels = []
            for (data, img_names) in data_loader:
                data = data.to(self.device)

                output = self.model(data)

                pred_score = nn.Sigmoid()(output[0]).data.cpu().numpy()#.reshape(1,-1)
                th = self.cfg['threshold']
                pred_score[pred_score>=th] = 1
                pred_score[pred_score<th] = 0
                pred = np.array(pred_score, dtype=np.float32)

                for i in range(len(pred)):
                    mask = pred[i][0]
                    res_dict[os.path.basename(img_names[i])] =  cv2.resize(mask, (512,512))
        return res_dict

   
    def evaluate(self, data_loader):
        self.model.eval()

        val_loss = 0
        val_mIOU = 0
        count = 0
        # pred_masks = []
        # true_masks = []

        with torch.no_grad():
            pres = []
            labels = []
            for (data, target, img_names) in data_loader:
                data, target = data.to(self.device), target.to(self.device)

                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    #print(output[0].shape, target.shape)
                    losses = self.loss_func(output[0], target) # sum up batch loss
                    val_loss += losses.item()

                #print(output.shape)
                #bs = target.shape[0]


                pre_mask = output[0].data.cpu().numpy()#.reshape(1,-1)
                target = target.data.cpu().numpy()#.reshape(1,-1)
                #print('\n',np.sum(target[0]))
                #print(target[0,0,-1])
                # print(pre_mask.shape, target.shape)
                # show = np.argmax(pre_mask[0], axis=0)
                # cv2.imwrite("eval_pre.png", np.reshape(show, (512,512)))
                # cv2.imwrite("eval_pre_show.png", np.reshape(show, (512,512))*20)
                target = np.array(target, dtype=np.int32)
                # print(pre_mask.shape, target.shape)
                # cv2.imwrite("eval_target.png", np.reshape(target, (512,512)))
                # cv2.imwrite("eval_target_show.png", np.reshape(target, (512,512))*20)
                mIOU = self.computeIOU(pre_mask,target)
      
                # print(mIOU, img_names)
                val_mIOU += mIOU
                count += 1
                # bb

        # ap10 = self.getAP10(pred_masks, true_masks, generate_gt=(epoch==0))

        val_loss /= count
        val_mIOU =  val_mIOU / count
        best_score = val_mIOU 


        print("[INFO] Eval val_loss: ",val_loss)
        print("[INFO] Eval val_mIOU: ",val_mIOU)
        print("[INFO] Eval best_score: ",best_score)


    def onTrainStart(self):
        

        self.early_stop_value = 0
        self.early_stop_dist = 0
        self.last_save_path = None

        self.earlystop = False
        self.best_epoch = 0

        # log
        self.log_time = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))


    def OHEM(self,output,target,score_th=0.5,area_th=0.002):
        sample_weights = []
        #print(output[0].shape,target.shape)
        pre = nn.Sigmoid()(output[0].detach())
        pre[pre>=score_th] = 1
        pre[pre<score_th] = 0
        #print(pre)

        mask = torch.abs(pre-target)
        #print(mask)
        diff = torch.sum(mask, dim=[1,2,3])
        #print(diff.shape)
        total_area = target.shape[-1]*target.shape[-2]
        #print(diff,total_area)
        area_ratio = diff/total_area
        # print(area_ratio)
        w = torch.ones(area_ratio.shape).to(area_ratio.device)
        pos_w = w*2
        neg_w = w*1
        # print(area_ratio>area_th, pos_w)
        sample_weights = torch.where(area_ratio>area_th,pos_w,neg_w)
        # print(sample_weights)
        # bb

        return sample_weights

    def onTrainStep(self,train_loader, epoch):
        
        self.model.train()

        count = 0
        batch_time = 0
        total_loss = 0
        loss1 = 0
        loss2 = 0
        loss3 = 0

        for batch_idx, (data, target, img_names) in enumerate(train_loader):
            # print(target.shape)
            # bb
            one_batch_time_start = time.time()

            target = target.to(self.device)

            data = data.to(self.device)

            with torch.cuda.amp.autocast():
                output = self.model(data)
                #[out, out16, out32, detail8]

                #all_linear2_params = torch.cat([x.view(-1) for x in model.model_feature._fc.parameters()])
                #l2_regularization = 0.0003 * torch.norm(all_linear2_params, 2)
                # sample_weights = None
                # if epoch>=20:
                #     sample_weights = self.OHEM(output,target,self.cfg['threshold'])

                loss = self.loss_func(output, target, epoch=epoch)# + l2_regularization.item()    
                count += 1

            total_loss += loss.item()
            if self.cfg['clip_gradient']:
                clipGradient(self.optimizer, self.cfg['clip_gradient'])


            self.optimizer.zero_grad()#把梯度置零
            # loss.backward() #计算梯度
            # self.optimizer.step() #更新参数
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

         
            #count+=target.shape[0]

            train_loss = total_loss/count
            # train_mIOU = mIOU/count

            one_batch_time = time.time() - one_batch_time_start
            batch_time+=one_batch_time
            eta = int((batch_time/(batch_idx+1))*(len(train_loader)-batch_idx-1))


            print_epoch = ''.join([' ']*(4-len(str(epoch+1))))+str(epoch+1)
            print_epoch_total = str(self.cfg['epochs'])+''.join([' ']*(4-len(str(self.cfg['epochs']))))

            log_interval = 10
            if batch_idx % log_interval== 0:
                print('\r',
                    '{}/{} [{}/{} ({:.0f}%)] - ETA: {}, loss: {:.4f}  LR: {:f}'.format(
                    print_epoch, print_epoch_total, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), 
                    datetime.timedelta(seconds=eta),
                    train_loss,
                    self.optimizer.param_groups[0]["lr"]), 
                    end="",flush=True)
            #break
        # show_img = cv2.merge([target[0][0]*0,target[0][0]*255,pred_score[0][0]*255])
        # cv2.imwrite("output/imgs/e%d.jpg" % epoch, show_img)


    def onTrainEnd(self):
        save_name = 'last_g%s.pth' % (self.cfg['GPU_ID'])
        self.last_save_path = os.path.join(self.cfg['save_dir'], save_name)
        self.modelSave(self.last_save_path)
        
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        if self.cfg["cfg_verbose"]:
            printDash()
            print(self.cfg)
            printDash()


    def computeIOU(self, pre_mask, target):
        #print(pre_mask.shape, target.shape)
        

        pre_mask = pre_mask.transpose(0,2,3,1)
        pre_mask = pre_mask.reshape(pre_mask.shape[0],-1,pre_mask.shape[-1])
        #print(x.shape)
        target = target.transpose(0,2,3,1)
        target = target.reshape(target.shape[0],-1)

        
        pre_mask = np.argmax(pre_mask, axis=-1)
        # print(pre_mask)
        # pre_mask = pre_mask.astype(np.uint8)
        # print(pre_mask)
        # print(target.shape, pre_mask.shape)
        # cv2.imwrite("gt.png", np.reshape(target,(512,512))*20)
        # cv2.imwrite("pre.png", np.reshape(pre_mask,(512,512))*20)
        mIOUs = []
        for bid in range(target.shape[0]):
            tp = np.sum((pre_mask[bid]==target[bid]) & (target[bid]>0))
            fp = np.sum((pre_mask[bid]!=target[bid]) & (target[bid]==0))
            #tn = np.sum((pre_mask[bid]==target[bid]) & (target==0))
            fn = np.sum((pre_mask[bid]!=target[bid]) & (target[bid]>0))
            #print(tp,fp,tn,fn)
            mIOU = 1*tp/(tp+fp+fn+1e-7)#+0.5*tn/(tn+fp+fn)
            #print("total ",mIOU)
            mIOUs.append(mIOU)

        # mIOUs = [[] for _ in range(self.cfg['class_number'])]
        # for bid in range(target.shape[0]):
        #     for i in range(1,self.cfg['class_number']):

        #         pre = pre_mask[bid]==i
        #         gt = target[bid]==i
        #         #print('-------',i, np.sum(pre), np.sum(gt))
        #         tp = np.sum((pre==gt) & (gt==1))
        #         fp = np.sum((pre!=gt) & (gt==0))
        #         fn = np.sum((pre!=gt) & (gt==1))
        #         #print(tp,fp,fn)
        #         mIOU = 1*tp/(tp+fp+fn+1e-7)#+0.5*tn/(tn+fp+fn)
        #         mIOUs.append(mIOU)
        #     # print(mIOUs, np.mean(mIOUs))
        #     # bb

        #     mIOUs.append(mIOU)
        
        #print(np.mean(mIOUs))



        # tp = np.sum((pre_mask==target) & (target>0))
        # fp = np.sum((pre_mask!=target) & (target==0))
        # #tn = np.sum((pre_mask==target) & (target==0))
        # fn = np.sum((pre_mask!=target) & (target>0))
        # #print(tp,fp,tn,fn)
        # mIOU = 1.0*tp/(tp+fp+fn+1e-7)#+0.5*tn/(tn+fp+fn)
        # print(mIOU)
        # bb
        return np.mean(mIOUs)

    def onValidation(self, val_loader, epoch):

        self.model.eval()
        self.val_loss = 0
        self.val_mIOU = 0
        count = 0
        # pred_masks = []
        # true_masks = []
        class_count = [0 for _ in range(self.cfg['class_number'])]
        hist = torch.zeros(self.cfg['class_number'], self.cfg['class_number']).cuda().detach()
        with torch.no_grad():
            pres = []
            labels = []
            for (data, target, img_names) in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    #print(output[0].shape, target.shape)
                    losses = self.loss_func(output, target) # sum up batch loss
                    self.val_loss += losses.item()

                #print(output.shape)
                #bs = target.shape[0]


                pre_mask = output[0].data.cpu().numpy()#.reshape(1,-1)
                preds = torch.argmax(output[0], dim=1)
                
                target_np = target.data.cpu().numpy()#.reshape(1,-1)
                #print('\n',np.sum(target[0]))
                #print(target[0,0,-1])
                # print(preds)
                #unique_values, indices_list = np.unique(preds, return_index=True)
                # print(unique_values)
                # print(target_np)
                #unique_values, indices_list = np.unique(target_np, return_index=True)
                # print(unique_values)
                # bb
                target_np = np.array(target_np, dtype=np.int32)
                target = target.squeeze(1)
                #print(target.shape) [8, 960, 544]
                for i in range(len(target_np)):
                    uniques = np.unique(target_np[i]).tolist()
                    for num in uniques:
                        class_count[num] += 1
                
                #bb
                keep = target != 255#self.ignore_label
                #mIOU = self.computeIOU(pre_mask,target)
                hist += torch.bincount(
                        target[keep] * self.cfg['class_number'] + preds[keep],
                        minlength=self.cfg['class_number'] ** 2
                        ).view(self.cfg['class_number'], self.cfg['class_number']).float()
                #print(IOUs)
                #self.val_mIOU += mIOU
                count += 1
                #bb
        print("class_count: ",class_count)
        # ap10 = self.getAP10(pred_masks, true_masks, generate_gt=(epoch==0))
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        ious[torch.isnan(ious)] = 0
        print(ious)
        miou = ious.mean()

        self.val_loss /= count
        self.val_mIOU =  miou#self.val_mIOU / count
        self.best_score = self.val_mIOU 

        
  
        print(' \n           [VAL] loss: {:.4}, mIOU: {:.5f} \n'.format(
            self.val_loss, self.val_mIOU))


        if self.cfg['warmup_epoch']:
            self.scheduler.step(epoch)
        else:
            if 'default' in self.cfg['scheduler']:
                self.scheduler.step(self.best_score)
            else:
                self.scheduler.step()


        self.checkpoint(epoch)
        self.earlyStop(epoch)

        


    def onTest(self):
        self.model.eval()
        
        #predict
        res_list = []
        with torch.no_grad():
            #end = time.time()
            for i, (inputs, target, img_names) in enumerate(data_loader):
                print("\r",str(i)+"/"+str(test_loader.__len__()),end="",flush=True)

                inputs = inputs.cuda()

                output = model(inputs)
                output = output.data.cpu().numpy()

                for i in range(output.shape[0]):

                    output_one = output[i][np.newaxis, :]
                    output_one = np.argmax(output_one)

                    res_list.append(output_one)
        return res_list



    def earlyStop(self, epoch):
        ### earlystop
        if self.best_score>self.early_stop_value:
            self.early_stop_value = self.best_score
            self.early_stop_dist = 0

        self.early_stop_dist+=1
        if self.early_stop_dist>self.cfg['early_stop_patient']:
            self.best_epoch = epoch-self.cfg['early_stop_patient']+1
            print("[INFO] Early Stop with patient %d , best is Epoch - %d :%f" % (self.cfg['early_stop_patient'],self.best_epoch,self.early_stop_value))
            self.earlystop = True
        if  epoch+1==self.cfg['epochs']:
            self.best_epoch = epoch-self.early_stop_dist+2
            print("[INFO] Finish trainging , best is Epoch - %d :%f" % (self.best_epoch,self.early_stop_value))
            self.earlystop = True

    def checkpoint(self, epoch):
        
        if self.best_score<=self.early_stop_value:
            if self.cfg['save_best_only']:
                pass
            else:
                save_name = '%s_e%d_%.5f.pth' % (self.cfg['model_name'],epoch+1,self.best_score)
                self.last_save_path = os.path.join(self.cfg['save_dir'], save_name)
                self.modelSave(self.last_save_path)
        else:
            if self.cfg['save_one_only']:
                if self.last_save_path is not None and os.path.exists(self.last_save_path):
                    os.remove(self.last_save_path)
            save_name = '%s_%s_e%d_%.5f.pth' % (self.cfg['model_name'],self.cfg['backbone'],epoch+1,self.best_score)
            self.last_save_path = os.path.join(self.cfg['save_dir'], save_name)
            self.modelSave(self.last_save_path)




    def modelLoad(self,model_path, data_parallel = False):
        ckpt = torch.load(model_path)
        
        ckpt = {k.replace('module.',''):v for k,v in ckpt.items()}
        self.model.load_state_dict(ckpt, strict=True)
        
        if data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def modelSave(self, save_name):
        torch.save(self.model.state_dict(), save_name)

    def toOnnx(self, save_name= "model.onnx"):
        dummy_input = torch.randn(1, 3, self.cfg['img_size'][0], self.cfg['img_size'][1]).to(self.device)

        torch.onnx.export(self.model, 
                        dummy_input, 
                        os.path.join(self.cfg['save_dir'],save_name), 
                        verbose=True)


