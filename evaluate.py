import os,argparse
import random
        
from fire import initFire, FireModel, FireRunner, FireData

from config import cfg




def main(cfg):

    cfg['batch_size'] = cfg['test_batch_size']
    initFire(cfg)


    model = FireModel(cfg)
    
    

    data = FireData(cfg)
    # data.showTrainData()
    # b
    
    train_loader,val_loader = data.getTrainValDataloader()


    runner = FireRunner(cfg, model)


    runner.modelLoad(cfg['model_path'])


    runner.evaluate(val_loader)



if __name__ == '__main__':
    main(cfg)