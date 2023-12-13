import os,argparse
import random
        
from fire import initFire, FireModel, FireRunner, FireData

from config import cfg
import pandas as pd




def main(cfg):

    cfg['test_path'] = "../data/train_with_seg/train"
    cfg['batch_size'] = 1

    initFire(cfg)

    model = FireModel(cfg)
    
    data = FireData(cfg)
    # data.showTrainData()
    # b

    data_loader = data.getTestDataloader()


    runner = FireRunner(cfg, model)

    #print(model)
    runner.modelLoad(cfg['model_path'])


    runner.predictShow(data_loader, "./output/trainval/")





if __name__ == '__main__':
    main(cfg)