
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fire.models.segmentation_models_pytorch as smp




JaccardLoss = smp.losses.JaccardLoss(mode='binary')
DiceLoss    = smp.losses.DiceLoss(mode='binary')
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss  = smp.losses.LovaszLoss(mode='binary', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='binary', log_loss=False)
FocalLoss    = smp.losses.FocalLoss(mode='binary')

###########################  loss

def labelSmooth(one_hot, label_smooth):
    return one_hot*(1-label_smooth)+label_smooth/one_hot.shape[1]


class CrossEntropyLossOneHot(nn.Module):
    def __init__(self):
        super(CrossEntropyLossOneHot, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, labels):
        return torch.mean(torch.sum(-labels * self.log_softmax(preds), -1))


class CrossEntropyLossV2(nn.Module):
    def __init__(self, label_smooth=0, class_weight=None):
        super().__init__()
        self.class_weight = class_weight 
        self.label_smooth = label_smooth
        self.epsilon = 1e-7
        
    def forward(self, x, y, label_smooth=0, gamma=0, sample_weights=None, sample_weight_img_names=None):

        #one_hot_label = F.one_hot(y, x.shape[1])
        one_hot_label = y
        if label_smooth:
            one_hot_label = labelSmooth(one_hot_label, label_smooth)

        #y_pred = F.log_softmax(x, dim=1)
        # equal below two lines
        y_softmax = F.softmax(x, 1)
        #print(y_softmax)
        y_softmax = torch.clamp(y_softmax, self.epsilon, 1.0-self.epsilon)# avoid nan
        y_softmaxlog = torch.log(y_softmax)

        # original CE loss
        loss = -one_hot_label * y_softmaxlog

        if class_weight:
            loss = loss*self.class_weight

        #focal loss gamma
        if gamma:
            loss = loss*((1-y_softmax)**gamma)

        loss = torch.mean(torch.sum(loss, -1))

        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, label_smooth=0, class_weight=None):
        super().__init__()
        self.class_weight = class_weight 
        self.label_smooth = label_smooth
        self.epsilon = 1e-7


    def forward_one(self, x, y, epoch=0, gamma=0, sample_weights=None):
        bs = x.shape[0]
        #求单类的loss
        x_sigmoid = torch.sigmoid(x)

        #loss2 = JaccardLoss(x_sigmoid,y)
        loss2 = DiceLoss(x_sigmoid,y)
        w2 = 1#.2
        # if epoch>20:
        #     w2=0.3
        loss3 = FocalLoss(x_sigmoid,y)
        w3 = 0#.3

        x_sigmoid = torch.reshape(x_sigmoid,(bs,-1,1))
        x_sigmoid_bg = 1-x_sigmoid
        #print(x_sigmoid,x_sigmoid2)
        x_sigmoid = torch.cat((x_sigmoid_bg,x_sigmoid),dim=-1)

        y = torch.reshape(y,(bs,-1,1))
        y_bg = 1-y
        #print(x_sigmoid,x_sigmoid2)
        y_onehot = torch.cat((y_bg,y),dim=-1)


        if self.label_smooth:
            y_onehot = labelSmooth(y_onehot, self.label_smooth)

        
        #print(y_softmax)
        x_sigmoid = torch.clamp(x_sigmoid, self.epsilon, 1.0-self.epsilon)# avoid nan
        x_sigmoidlog = torch.log(x_sigmoid)

        # original CE loss
        loss = -y_onehot * x_sigmoidlog

        if self.class_weight is not None:
            #print(loss.shape)
            loss = loss*self.class_weight
        
        if gamma:
            loss = loss*(y*(1-x_sigmoid)**gamma+(1-y)*(x_sigmoid)**gamma)

        if sample_weights is not None:
            sample_weights = sample_weights.view(len(sample_weights),1,1)
            loss = loss*sample_weights
        # print(torch.sum(loss, dim=[1,2,3]))
        # bb
        loss = torch.mean(torch.mean(loss, dim=[1,2]))
        loss_all = loss+loss2*w2+loss3*w3
        return loss_all

        
    def forward(self, x, y, epoch=0, gamma=0, sample_weights=None):
        x = x[0]
        n,c,h,w = x.shape
        #print(x.shape, y.shape)

        x = x.permute(0,2,3,1)
        x = x.reshape(x.shape[0],-1,x.shape[-1])
        #print(x.shape)
        y = y.permute(0,2,3,1)
        y = y.reshape(y.shape[0],-1)
        # y = y.reshape(-1)
        #print(y.shape)
        y = F.one_hot(y, num_classes=8)
        #print(y.shape)
        #bb

        x = x.reshape(-1,x.shape[-1])
        one_hot_label = y.reshape(-1,y.shape[-1])

        if self.label_smooth:
            one_hot_label = labelSmooth(one_hot_label, self.label_smooth)

        #y_pred = F.log_softmax(x, dim=1)
        # equal below two lines
        y_softmax = F.softmax(x, 1)

        y_softmax = torch.clamp(y_softmax, self.epsilon, 1.0-self.epsilon)# avoid nan
        y_softmaxlog = torch.log(y_softmax)

        # original CE loss
        loss = -one_hot_label * y_softmaxlog

        if self.class_weight is not None:
            loss = loss*self.class_weight

        if gamma:
            loss = loss*((1-y_softmax)**gamma)

        loss = torch.mean(torch.sum(loss, -1))

        return loss

def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()

def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)   # reshape 为向量
    ones = torch.sparse.torch.eye(N).cuda()
    ones = ones.index_select(0, label.long())   # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)

def get_boundary(gtmasks):

    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32, device=gtmasks.device).reshape(1, 1, 3, 3).requires_grad_(False)
    # boundary_logits = boundary_logits.unsqueeze(1)
    boundary_targets = F.conv2d(gtmasks.unsqueeze(1), laplacian_kernel, padding=1)
    boundary_targets = boundary_targets.clamp(min=0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0
    return boundary_targets

class DetailAggregateLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DetailAggregateLoss, self).__init__()
        
        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)
        
        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6./10], [3./10], [1./10]],
            dtype=torch.float32).reshape(1, 3, 1, 1).type(torch.cuda.FloatTensor))

    def forward(self, boundary_logits, gtmasks):

        # boundary_logits = boundary_logits.unsqueeze(1)
        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)
        
        boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=8, padding=1)
        boundary_targets_x8 = boundary_targets_x8.clamp(min=0)
    
        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')
        
        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0
        
        
        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0
       
        
        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0
        
        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim=1)
        
        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)

        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0
        
        
        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)
        
        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid)
        dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boudary_targets_pyramid)
        return bce_loss,  dice_loss

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
                nowd_params += list(module.parameters())
        return nowd_params


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        #print(logits)
        loss, _ = torch.sort(loss, descending=True)
        #print(loss)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class STDCLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.class_weight = class_weight 
        # self.label_smooth = label_smooth
        self.epsilon = 1e-7

        ignore_idx = 255
        score_thres = 0.7
        n_min = cfg['batch_size']*cfg['img_size'][0]*cfg['img_size'][1]//16
        self.criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
        self.criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
        self.criteria_32 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
        self.boundary_loss_func = DetailAggregateLoss()

    def forward(self, x, y, epoch=0, gamma=0, sample_weights=None):
        out, out16, out32, detail8 = x 
        lb = torch.squeeze(y, 1)
        

        lossp = self.criteria_p(out, lb)
        loss2 = self.criteria_16(out16, lb)
        loss3 = self.criteria_32(out32, lb)

        boundery_bce_loss = 0.
        boundery_dice_loss = 0.

        boundery_bce_loss8,  boundery_dice_loss8 = self.boundary_loss_func(detail8, lb)
        boundery_bce_loss += boundery_bce_loss8
        boundery_dice_loss += boundery_dice_loss8


        loss = lossp + loss2 + loss3 + boundery_bce_loss + boundery_dice_loss
   

        return loss


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))
        
    def forward(self, x, y):

        # loss = criterion(x.reshape(-1, 512, 512, 1), y[:, :1, :, :].float().reshape(-1, 512, 512, 1))
        loss = self.criterion(x, y)
        return loss

# class FocalLoss(nn.Module):
#     def __init__(self, label_smooth=0, gamma = 0., weight=None):
#         super().__init__()
#         self.gamma = gamma
#         self.weight = weight # means alpha
#         self.epsilon = 1e-7
#         self.label_smooth = label_smooth

        
#     def forward(self, x, y, sample_weights=0, sample_weight_img_names=None):

#         if len(y.shape) == 1:
#             #
#             one_hot_label = F.one_hot(y, x.shape[1])

#             if self.label_smooth:
#                 one_hot_label = labelSmooth(one_hot_label, self.label_smooth)

#             if sample_weights>0 and sample_weights is not None:
#                 #print(sample_weight_img_names)
#                 weigths = [sample_weights  if 'yxboard' in img_name  else 1 for img_name in sample_weight_img_names] 
#                 weigths = torch.DoubleTensor(weigths).reshape((len(weigths),1)).to(x.device)
#                 #print(weigths, weigths.shape)
#                 #print(one_hot_label, one_hot_label.shape)
#                 one_hot_label = one_hot_label*weigths
#                 #print(one_hot_label)
#                 #b
#         else:
#             one_hot_label = y


#         #y_pred = F.log_softmax(x, dim=1)
#         # equal below two lines
#         y_softmax = F.softmax(x, 1)
#         #print(y_softmax)
#         y_softmax = torch.clamp(y_softmax, self.epsilon, 1.0-self.epsilon)# avoid nan
#         y_softmaxlog = torch.log(y_softmax)

#         #print(y_softmaxlog)
#         # original CE loss
#         loss = -one_hot_label * y_softmaxlog
#         #loss = 1 * torch.abs(one_hot_label-y_softmax)#my new CE..ok its L1...

#         # print(one_hot_label)
#         # print(y_softmax)
#         # print(one_hot_label-y_softmax)
#         # print(torch.abs(y-y_softmax))
#         #print(loss)
        
#         # gamma
#         loss = loss*((torch.abs(one_hot_label-y_softmax))**self.gamma)
#         # print(loss)

#         # alpha
#         if self.weight is not None:
#             loss = self.weight*loss

#         loss = torch.mean(torch.sum(loss, -1))
#         return loss





if __name__ == '__main__':



    device = torch.device("cpu")

    #x = torch.randn(2,2)
    x = torch.tensor([[0.1,0.7,0.2]])
    y = torch.tensor([1])
    print(x)

    loss_func = torch.nn.CrossEntropyLoss().to(device)
    loss = loss_func(x,y)
    print("loss1: ",loss)

    # loss_func = Focalloss().to(device)
    # loss = loss_func(x,y)
    # print("loss2: ",loss)
    

    weight_loss = torch.DoubleTensor([1,1,1]).to(device)
    loss_func = FocalLoss(gamma=0, weight=weight_loss).to(device)
    loss = loss_func(x,y)
    print("loss3: ",loss)
    

    # weight_loss = torch.DoubleTensor([2,1]).to(device)
    # loss_func = Focalloss(gamma=0.2, weight=weight_loss).to(device)
    # loss = loss_func(x,y)
    # print("loss4: ",loss)