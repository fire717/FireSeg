# @https://github.com/fire717/Fire

cfg = {
    ### Global Set
    "model_name": "unet",    #unetplusplus  STDC
    "backbone": "efficientnet-b0",  #efficientnet-b3 STDCNet1446  STDCNet813
    'GPU_ID': '0',

    "class_number": 8,
    "random_seed":42,
    "cfg_verbose":True,
    "num_workers":4,


    ### Train Setting
    'train_path':"../data/",
    'pretrained':"imagenet", # "imagenet" local path or ''
    #STDCNet1446_76.47.tar
    #../STDCNet813M_73.91.tar"

    'try_to_train_items': 0,   # 0 means all, or run part(200 e.g.) for bug test
    'save_best_only': True,  #only save model if better than before
    'save_one_only':True,    #only save one best model (will del model before)
    "save_dir": "output/",
    'metrics': ['acc'], # default is acc,  can add F1  ...
    "loss": 'CE', # CE, CEV2-0.5, Focalloss-1 ...STDCLoss

    'show_data':False,

    ### Train Hyperparameters
    "img_size": [544, 960], # [h, w] 896
    'learning_rate':0.001,
    'batch_size':8,
    'epochs':35,
    'optimizer':'Adam',  #Adam  SGD AdaBelief Ranger
    'scheduler':'SGDR-35-1', #default  SGDR-5-2    step-4-0.8
    'threshold': 0.5,

    'warmup_epoch':0, # 
    'weight_decay' : 0,#0.0001,
    "k_flod":5,
    'val_fold':0,
    'early_stop_patient':20,

    'use_distill':0,
    'label_smooth':0,
    # 'checkpoint':None,
    'class_weight': None,#s[1.4, 0.78], # None [1, 1]
    'clip_gradient': 0,#1,       # 0
    'freeze_nonlinear_epoch':0,

    'dropout':0.5, #before last_linear

    'mixup':False,
    'cutmix':False,
    'sample_weights':None,


    ### Test
    'model_path':'output/upernet_convnext_tiny_e24_0.49296.pth',#test model

    'test_path':"../img",#test without label, just show img result
    'test_batch_size':16,
}
