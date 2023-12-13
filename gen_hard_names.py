import os,argparse
import random
        
from fire import initFire, FireModel, FireRunner, FireData

from config import cfg
import pandas as pd




def main(cfg):

    cfg['test_path'] = "../data/chusai/train"
    cfg['batch_size'] = 1

    initFire(cfg)

    model = FireModel(cfg)
    
    data = FireData(cfg)
    # data.showTrainData()
    # b

    train_loader,val_loader = data.getTrainValDataloader()


    runner = FireRunner(cfg, model)

    #print(model)
    runner.modelLoad(cfg['model_path'])


    runner.predictHard(train_loader, "./output/hard_train.txt")
    runner.predictHard(val_loader, "./output/hard_val.txt")




if __name__ == '__main__':
    main(cfg)