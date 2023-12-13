import torch
import torch.nn as nn
import torch.nn.functional as F

from fire.models.segmentation_models_pytorch.decoders.pspnet.decoder import (
    PSPModule,
)


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(policy))
        self.policy = policy

    def forward(self, x):
        if self.policy == "add":
            return sum(x)
        elif self.policy == "cat":
            return torch.cat(x, dim=1)
        else:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy))

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class UperNetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        encoder_depth=5,
        pyramid_channels=256,
        segmentation_channels=128,
        dropout=0.2,
        merge_policy="cat",
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))


        



        self.ppm = PSPModule(
            in_channels=encoder_channels[-1],
            sizes=(1, 2, 3, 6),
            use_bathcnorm=True,
        )

        
        #print(encoder_channels) (3, 128, 256, 512, 1024) conv-b
        

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[: encoder_depth + 1]

        #print(encoder_channels) (1024, 512, 256, 128, 3)
        
        # self.p5 = nn.Conv2d(encoder_channels[0]*2, pyramid_channels, kernel_size=1)
        # self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        # self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        # self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        # self.seg_blocks = nn.ModuleList(
        #     [
        #         SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
        #         for n_upsamples in [3, 2, 1, 0]
        #     ]
        # )

        # self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=0.3, inplace=True)



        # fc_dim=4096
        # pool_scales=(1, 2, 3, 6)
        fpn_inplanes = encoder_channels[:4] #(384, 136, 48, 32)

        fpn_dim=256
        self.ppm_last_conv = conv3x3_bn_relu(fpn_inplanes[0]*2, fpn_dim, 1)
        #print("self.encoder.out_channels: ",self.encoder.out_channels) (6, 40, 32, 48, 136, 384)
        
        #print("fpn_inplanes: ",fpn_inplanes) 40, 32, 48, 136, 384
        # FPN Module
        self.fpn_in = []
        for fpn_inplane in reversed(fpn_inplanes[1:]): # skip the top layer
            print("fpn_inplane: ",fpn_inplane)
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1): # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.conv_fusion = conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, 128, 1)

        # self.object_head = nn.Sequential(
        #     conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
        #     nn.Conv2d(fpn_dim, 1, kernel_size=1, bias=True)
        # )

    def forward(self, *features):
        #print([x.shape for x in features])
        # [torch.Size([8, 6, 896, 896]), torch.Size([8, 40, 448, 448]), torch.Size([8, 32, 224, 224]), 
        # torch.Size([8, 48, 112, 112]), torch.Size([8, 136, 56, 56]), torch.Size([8, 384, 28, 28])]
        features = features[-4:]
        c2, c3, c4, c5 = features#[-4:]
        #print(c2.shape,c3.shape,c4.shape,c5.shape)
        #[8, 32, 224, 224]) torch.Size([8, 48, 112, 112]) torch.Size([8, 136, 56, 56]) torch.Size([8, 384, 28, 28]
        p5 = self.ppm(c5)
        #print(p6.shape) [8, 768, 28, 28
        p5 = self.ppm_last_conv(p5)

        fpn_feature_list = [p5]
        #print(self.fpn_in)
        for i in reversed(range(len(features) - 1)):
            conv_x = features[i]
            #print(i, features[i].shape)
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            p5 = F.interpolate(
                p5, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            #print(features[1+i].shape, conv_x.shape, p5.shape)
            p5 = conv_x + p5

            fpn_feature_list.append(self.fpn_out[i](p5))
        fpn_feature_list.reverse() # [P2 - P5]

        output_size = fpn_feature_list[0].size()[2:]
        # print(output_size)
        # bb
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(F.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        x = torch.cat(fusion_list, 1)
        x = self.dropout(x)
        #print(fusion_out.shape)
        x = self.conv_fusion(x)
        #print(x.shape)
        # x = self.object_head(x)
        # print(x.shape)

        #p5 = self.p5(pp)
        #print(p5.shape)  #[8, 256, 28, 28]
        # p4 = self.p4(p5, c4)
        # #print(p4.shape) [8, 256, 56, 56]
        # p3 = self.p3(p4, c3)
        # #print(p3.shape) [8, 256, 112, 112]
        # p2 = self.p2(p3, c2)
        # #print(p2.shape)  [8, 256, 224, 224]
        # feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        # x = self.merge(feature_pyramid)
        # #print(x.shape) [8, 128, 224, 224]
        # x = self.conv_fusion(x)
        # x = self.dropout(x)
        #print(x.shape) [8, 128, 224, 224]
       # bb
        return x


        #官方：fpn输出先插值放大，然后cat，然后经过conv_fusion，然后经过head（conv-bn-relu, 1x1conv）
        #smp的head：[conv2d, upsampling, activation]
        #第三方：fpn输出先插值放大，然后cat，然后经过conv_fusion，然后经过seg（conv-bn-relu, 1x1conv），然后经过out（插值放大，conv）