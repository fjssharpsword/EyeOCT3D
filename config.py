import os

config = {
            'CKPT_PATH': '/data/pycode/EyeOCT3D/ckpt/',
            'log_path':  '/data/pycode/EyeOCT3D/log/',
            'img_path': '/data/pycode/EyeOCT3D/imgs/',
            'CUDA_VISIBLE_DEVICES': "0,1,2,3,4,5,6,7",
            'CUBE_SIZE': [100, 100, 160],  #D*H*W
            'MAX_EPOCHS': 50,
            'CODE_SIZE': 8,
            'BATCH_SIZE': 24
         } 



