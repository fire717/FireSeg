from typing import Optional, Union, List
import torch.nn as nn
import torch.nn.functional as F
from fire.models.segmentation_models_pytorch.encoders import get_encoder
from fire.models.segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from fire.models.segmentation_models_pytorch.decoders.pspnet.decoder import (
    PSPModule,
)

from .decoder import UperNetDecoder



class UperNet(SegmentationModel):
    def __init__(self, 
                encoder_name: str = "resnet34",
                encoder_depth: int = 5,
                encoder_weights: Optional[str] = "imagenet",
                decoder_pyramid_channels: int = 256,
                decoder_segmentation_channels: int = 128,
                decoder_merge_policy: str = "add",
                decoder_dropout: float = 0.2,
                in_channels: int = 3,
                classes: int = 1,
                activation: Optional[str] = None,
                upsampling: int = 4,
                aux_params: Optional[dict] = None,):
        super().__init__()

        # validate input params
        # if encoder_name.startswith("mit_b") and encoder_depth != 5:
        #     raise ValueError("Encoder {} support only encoder_depth=5".format(encoder_name))

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UperNetDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )
        # self.segmentation_head = SegmentationHead(
        #     in_channels=decoder_channels[-1],
        #     out_channels=classes,
        #     activation=activation,
        #     kernel_size=3,
        # )

        self.neck1 = nn.Sequential(
                         nn.Conv2d(self.encoder.out_channels[1]*2, self.encoder.out_channels[1], 1, 1, 0, bias=False),
                         nn.BatchNorm2d(self.encoder.out_channels[1]),
                         nn.ReLU(inplace=True),
                         nn.Dropout2d(0.1),
                         )
        self.neck2 = nn.Sequential(
                         nn.Conv2d(self.encoder.out_channels[2]*2, self.encoder.out_channels[2], 1, 1, 0, bias=False),
                         nn.BatchNorm2d(self.encoder.out_channels[2]),
                         nn.ReLU(inplace=True),
                         nn.Dropout2d(0.1),
                         )
        self.neck3 = nn.Sequential(
                         nn.Conv2d(self.encoder.out_channels[3]*2, self.encoder.out_channels[3], 1, 1, 0, bias=False),
                         nn.BatchNorm2d(self.encoder.out_channels[3]),
                         nn.ReLU(inplace=True),
                         nn.Dropout2d(0.1),
                         )
        self.neck4 = nn.Sequential(
                         nn.Conv2d(self.encoder.out_channels[4]*2, self.encoder.out_channels[4], 1, 1, 0, bias=False),
                         nn.BatchNorm2d(self.encoder.out_channels[4]),
                         nn.ReLU(inplace=True),
                         nn.Dropout2d(0.1),
                         )
        self.necks = [self.neck1, self.neck2, self.neck3, self.neck4]

        if len(self.encoder.out_channels)==6:
            self.neck5 = nn.Sequential(
                             nn.Conv2d(self.encoder.out_channels[5]*2, self.encoder.out_channels[5], 1, 1, 0, bias=False),
                             nn.BatchNorm2d(self.encoder.out_channels[5]),
                             nn.ReLU(inplace=True),
                             nn.Dropout2d(0.1),
                             )
            self.necks.append(self.neck5)


        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "upernet-{}".format(encoder_name)
        self.initialize()