import os

config = {
            'CKPT_PATH': '/data/pycode/EyeOCT3D/ckpt/',
            'log_path':  '/data/pycode/EyeOCT3D/log/',
            'img_path': '/data/pycode/EyeOCT3D/imgs/',
            'CUDA_VISIBLE_DEVICES': "0,1,2,3,4,5,6,7",
            'CUBE_SIZE': [80, 80, 80],  #D*H*W
            'MAX_EPOCHS': 20,
            'CODE_SIZE': 64,
            'BATCH_SIZE': 80
         } 



