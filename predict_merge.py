import os,argparse
import random
        
from fire import initFire, FireModel, FireRunner, FireData

from config import cfg
import pandas as pd
import numpy as np
import cv2


def predictMerge(cfg):
    initFire(cfg)


    model = FireModel(cfg)
    data = FireData(cfg)

    test_loader = data.getTestDataloader()
    runner1 = FireRunner(cfg, model)
    runner1.modelLoad('output/unetplusplus_efficientnet-b3_e29_0.77840.pth')
    print("load model1, start running.")
    res_dict1 = runner1.predictRaw(test_loader)
    print(len(res_dict1))


    cfg['backbone'] = "efficientnet-b2"
    cfg["img_size"] = [1024,1024]
    model = FireModel(cfg)
    test_loader = data.getTestDataloader()
    runner2 = FireRunner(cfg, model)
    runner2.modelLoad('output/unetplusplus_efficientnet-b2_e29_0.88015.pth')
    print("load model2, start running.")
    res_dict2 = runner2.predictRaw(test_loader)


    cfg['backbone'] = "efficientnet-b4"
    cfg["img_size"] = [640,640]
    model = FireModel(cfg)
    test_loader = data.getTestDataloader()
    runner3 = FireRunner(cfg, model)
    runner3.modelLoad('output/unetplusplus_efficientnet-b4_e29_0.90489.pth')
    print("load model3, start running.")
    res_dict3 = runner3.predictRaw(test_loader)

    # test_loader = data.getTestDataloader()
    # runner4 = FireRunner(cfg, model)
    # runner4.modelLoad('output/unetplusplus_efficientnet-b3_e28_0.78753.pth')
    # print("load model4, start running.")
    # res_dict4 = runner4.predictRaw(test_loader)

    # test_loader = data.getTestDataloader()
    # runner5 = FireRunner(cfg, model)
    # runner5.modelLoad('output/unetplusplus_efficientnet-b3_e27_0.77918.pth')
    # print("load model5, start running.")
    # res_dict5 = runner5.predictRaw(test_loader)


    res_dict = {}
    for k,v in res_dict1.items():
        #print(k,v)
        # v1 = (v+res_dict2[k]+res_dict3[k]+res_dict4[k]+res_dict5[k])/5
        v1 = (v+res_dict2[k]+res_dict3[k])/3
        th = 0.5
        v1[v1>=th] = 1
        v1[v1<th] = 0
        #pred = np.array(pred_score, dtype=np.float32)
        res_dict[k] = v1
    
    res_list = sorted(res_dict.items(), key = lambda kv: int(kv[0].split("_")[-1].split('.')[0]))
    # print(len(res_list), res_list[0])


    for res in res_list:
        save_path = os.path.join("output/submit", os.path.basename(res[0])[:-3]+'png')
        mask = res[1]
        #mask = mask[:,:,np.newaxis]
        cv2.imwrite(save_path, mask)




def main(cfg):

    predictMerge(cfg)


    



if __name__ == '__main__':
    main(cfg)