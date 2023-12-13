import os,argparse
import random
        
from fire import initFire, FireModel, FireRunner, FireData

from config import cfg
import pandas as pd




def main(cfg):



    initFire(cfg)
    model = FireModel(cfg)
    data = FireData(cfg)
    # data.showTrainData()
    # b
    train_loader,val_loader = data.getTrainValDataloader()
    runner = FireRunner(cfg, model)
    runner.modelLoad('output/unetplusplus_efficientnet-b3_e29_0.77840.pth')
    runner.genDistill(val_loader, "../data/chusai/train/label2/")


    cfg['val_fold'] = 1
    initFire(cfg)
    model = FireModel(cfg)
    data = FireData(cfg)
    train_loader,val_loader = data.getTrainValDataloader()
    runner = FireRunner(cfg, model)
    runner.modelLoad('output/unetplusplus_efficientnet-b3_e28_0.77740.pth')
    runner.genDistill(val_loader, "../data/chusai/train/label2/")


    cfg['val_fold'] = 2
    initFire(cfg)
    model = FireModel(cfg)
    data = FireData(cfg)
    train_loader,val_loader = data.getTrainValDataloader()
    runner = FireRunner(cfg, model)
    runner.modelLoad('output/unetplusplus_efficientnet-b3_e29_0.77904.pth')
    runner.genDistill(val_loader, "../data/chusai/train/label2/")


    cfg['val_fold'] = 3
    initFire(cfg)
    model = FireModel(cfg)
    data = FireData(cfg)
    train_loader,val_loader = data.getTrainValDataloader()
    runner = FireRunner(cfg, model)
    runner.modelLoad('output/unetplusplus_efficientnet-b3_e28_0.78753.pth')
    runner.genDistill(val_loader, "../data/chusai/train/label2/")


    cfg['val_fold'] = 4
    initFire(cfg)
    model = FireModel(cfg)
    data = FireData(cfg)
    train_loader,val_loader = data.getTrainValDataloader()
    runner = FireRunner(cfg, model)
    runner.modelLoad('output/unetplusplus_efficientnet-b3_e27_0.77918.pth')
    runner.genDistill(val_loader, "../data/chusai/train/label2/")


if __name__ == '__main__':
    main(cfg)