import torch
import torch.nn as nn

import pretrainedmodels



# from fire.models.myefficientnet_pytorch import EfficientNet
# from fire.models.convnext import convnext_tiny,convnext_small,convnext_base,convnext_large
import fire.models.segmentation_models_pytorch as smp
from fire.models.fastscnn import FastSCNN
from fire.models.STDC.model_stages import BiSeNet

import torchvision

class FireModel(nn.Module):
    def __init__(self, cfg):
        super(FireModel, self).__init__()

        self.cfg = cfg
        
        self.pretrainedModel()
        
        self.changeModelStructure()
        


    def pretrainedModel(self):


        ### Create model
        if self.cfg['model_name']=="STDC":
            ignore_idx = 255
            self.pretrain_model = BiSeNet(backbone=self.cfg['backbone'], 
                                n_classes=self.cfg['class_number'], 
                                pretrain_model=self.cfg['pretrained'], 
                                use_boundary_2=False, 
                                use_boundary_4=False, 
                                use_boundary_8=True, 
                                use_boundary_16=False, 
                                use_conv_last=False)
            # print(self.pretrain_model)
            # bb
      


        elif self.cfg['model_name']=="unet":
          
            if "convnext" in self.cfg['backbone']:
                self.pretrain_model = smp.Unet(
                    encoder_name=self.cfg['backbone'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights=self.cfg['pretrained'],     # use `imagenet` pre-trained weights for encoder initialization
                    encoder_depth=4,
                    decoder_channels=( 128, 64, 32, 16),
                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    classes=self.cfg['class_number'],                      # model output channels (number of classes in your dataset)
                )
            else:
                self.pretrain_model = smp.Unet(
                    encoder_name=self.cfg['backbone'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights=self.cfg['pretrained'],     # use `imagenet` pre-trained weights for encoder initialization
                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    classes=self.cfg['class_number'],                      # model output channels (number of classes in your dataset)
                )
            # if self.cfg['pretrained']:
            #     state_dict = torch.load(self.cfg['pretrained'])
            #     self.pretrain_model.load_state_dict(state_dict, strict=True)
        elif self.cfg['model_name']=="upernet":
          
            
            if "convnext" in self.cfg['backbone']:
                self.pretrain_model = smp.UperNet(
                    encoder_name=self.cfg['backbone'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights=self.cfg['pretrained'],     # use `imagenet` pre-trained weights for encoder initialization
                    encoder_depth=4,
                    # decoder_channels=( 128, 64, 32, 16),
                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    classes=self.cfg['class_number'],                      # model output channels (number of classes in your dataset)
                )
            else:
                self.pretrain_model = smp.UperNet(
                    encoder_name=self.cfg['backbone'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights=self.cfg['pretrained'],     # use `imagenet` pre-trained weights for encoder initialization
                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    classes=self.cfg['class_number'],                      # model output channels (number of classes in your dataset)
                )

        elif self.cfg['model_name']=="unetplusplus":
          
            aux_params=dict(
                pooling='avg',             # one of 'avg', 'max'
                dropout=0.5,               # dropout ratio, default is None
                activation='sigmoid',      # activation function, default is None
                classes=1,                 # define number of output labels
            )

            if "convnext" in self.cfg['backbone']:
                self.pretrain_model = smp.UnetPlusPlus(
                    encoder_name=self.cfg['backbone'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights=self.cfg['pretrained'],     # use `imagenet` pre-trained weights for encoder initialization
                    encoder_depth=4,
                    decoder_channels=( 128, 64, 32, 16),
                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    classes=self.cfg['class_number'],                      # model output channels (number of classes in your dataset)
                )
            else:
                self.pretrain_model = smp.UnetPlusPlus(
                    encoder_name=self.cfg['backbone'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights=self.cfg['pretrained'],     # use `imagenet` pre-trained weights for encoder initialization
                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    classes=self.cfg['class_number'],                      # model output channels (number of classes in your dataset)
                    #aux_params=aux_params
                    )
            
        elif self.cfg['model_name']=="manet":
          
            self.pretrain_model = smp.MAnet(
                encoder_name=self.cfg['backbone'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=self.cfg['pretrained'],     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=6,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.cfg['class_number'],                      # model output channels (number of classes in your dataset)
            )
        
        elif self.cfg['model_name']=="linknet":
          
            self.pretrain_model = smp.Linknet(
                encoder_name=self.cfg['backbone'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=self.cfg['pretrained'],     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=6,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.cfg['class_number'],                      # model output channels (number of classes in your dataset)
            )

        elif self.cfg['model_name']=="fpn":
          
            self.pretrain_model = smp.FPN(
                encoder_name=self.cfg['backbone'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=self.cfg['pretrained'],     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=6,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.cfg['class_number'],                      # model output channels (number of classes in your dataset)
            )

        elif self.cfg['model_name']=="pspnet":
          
            self.pretrain_model = smp.PSPNet(
                encoder_name=self.cfg['backbone'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=self.cfg['pretrained'],     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=6,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.cfg['class_number'],                      # model output channels (number of classes in your dataset)
            )

        elif self.cfg['model_name']=="pan":
          
            self.pretrain_model = smp.PAN(
                encoder_name=self.cfg['backbone'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=self.cfg['pretrained'],     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=6,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.cfg['class_number'],                      # model output channels (number of classes in your dataset)
            )

        elif self.cfg['model_name']=="deeplabv3":
            
            if "convnext" in self.cfg['backbone']:
                self.pretrain_model = smp.DeepLabV3(
                    encoder_name=self.cfg['backbone'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights=self.cfg['pretrained'],     # use `imagenet` pre-trained weights for encoder initialization
                    encoder_depth=4,
                    in_channels=6,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    classes=self.cfg['class_number'],                      # model output channels (number of classes in your dataset)
                )
            else:
                self.pretrain_model = smp.DeepLabV3(
                    encoder_name=self.cfg['backbone'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights=self.cfg['pretrained'],     # use `imagenet` pre-trained weights for encoder initialization
                    in_channels=6,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    classes=self.cfg['class_number'],                      # model output channels (number of classes in your dataset)
                )

        elif self.cfg['model_name']=="deeplabv3plus":
            self.pretrain_model = smp.DeepLabV3Plus(
                encoder_name=self.cfg['backbone'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=self.cfg['pretrained'],     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=6,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.cfg['class_number'],                      # model output channels (number of classes in your dataset)
            )

        elif self.cfg['model_name']=="FastSCNN":
            self.pretrain_model = FastSCNN(in_channels=3, 
                num_classes=self.cfg['class_number'])
            ckpt = torch.load("../1024_2048_model.pth")
            # ckpt = {k.replace('module.',''):v for k,v in ckpt.items()}
            ckpt.pop('classifier.conv.weight')
            ckpt.pop('classifier.conv.bias')
            self.pretrain_model.load_state_dict(ckpt, strict=False)

        else:
            raise Exception("[ERROR] Unknown model_name: ",self.cfg['model_name'])


    def changeModelStructure(self):
        ### Change model
        pass
        # if self.cfg['model_name']=='unet':
        #     pass
        # elif self.cfg['model_name']=="unetplusplus":
        #     pass

        # else:
        #     raise Exception("[ERROR] Unknown model_name: ",self.cfg['model_name'])


    def forward(self, img):        
        if self.cfg['model_name']=="STDC":
            out, out16, out32, detail8 = self.pretrain_model(img)
            #(not use_boundary_2) and (not use_boundary_4) and use_boundary_8
            #print(out.shape, out16.shape, out32.shape, detail8.shape)
            #[14, 20, 512, 512]) torch.Size([14, 20, 512, 512]) torch.Size([14, 20, 512, 512]) torch.Size([14, 1, 64, 64]
            #bb
            # out = self.avgpool(out)
            # out = out.view(out.size(0), -1)
            # out1 = self.head1(out)

            out = [out, out16, out32, detail8]

        elif self.cfg['model_name']=="unet":
            out1 = self.pretrain_model(img)
            # out = self.avgpool(out)
            # out = out.view(out.size(0), -1)
            # out1 = self.head1(out)

            out = [out1]

        elif self.cfg['model_name']=="upernet":
            out1 = self.pretrain_model(img)
            # out = self.avgpool(out)
            # out = out.view(out.size(0), -1)
            # out1 = self.head1(out)

            out = [out1]

        elif self.cfg['model_name']=="unetplusplus":
            out1 = self.pretrain_model(img)
            # out = self.avgpool(out)
            # out = out.view(out.size(0), -1)
            # out1 = self.head1(out)

            out = [out1]


        elif self.cfg['model_name']=="manet":
            out1 = self.pretrain_model(img)
            # out = self.avgpool(out)
            # out = out.view(out.size(0), -1)
            # out1 = self.head1(out)

            out = [out1]

        elif self.cfg['model_name']=="linknet":
            out1 = self.pretrain_model(img)
            # out = self.avgpool(out)
            # out = out.view(out.size(0), -1)
            # out1 = self.head1(out)

            out = [out1]

        elif self.cfg['model_name']=="fpn":
            out1 = self.pretrain_model(img)
            # out = self.avgpool(out)
            # out = out.view(out.size(0), -1)
            # out1 = self.head1(out)

            out = [out1]

        elif self.cfg['model_name']=="pspnet":
            out1 = self.pretrain_model(img)
            # out = self.avgpool(out)
            # out = out.view(out.size(0), -1)
            # out1 = self.head1(out)

            out = [out1]

        elif self.cfg['model_name']=="pan":
            out1 = self.pretrain_model(img)
            # out = self.avgpool(out)
            # out = out.view(out.size(0), -1)
            # out1 = self.head1(out)

            out = [out1]

        elif self.cfg['model_name']=="deeplabv3":
            out1 = self.pretrain_model(img)
            # out = self.avgpool(out)
            # out = out.view(out.size(0), -1)
            # out1 = self.head1(out)

            out = [out1]

        elif self.cfg['model_name']=="deeplabv3plus":
            out1 = self.pretrain_model(img)
            # out = self.avgpool(out)
            # out = out.view(out.size(0), -1)
            # out1 = self.head1(out)

            out = [out1]

        elif self.cfg['model_name']=="FastSCNN":
            out1 = self.pretrain_model(img)
            # out = self.avgpool(out)
            # out = out.view(out.size(0), -1)
            # out1 = self.head1(out)

            out = [out1]
        # [Add new model here]
        # elif self.cfg['model_name']=="xxx":
        #     pass

        else:
            raise Exception("[ERROR] Unknown model_name: ",self.cfg['model_name'])

        return out


