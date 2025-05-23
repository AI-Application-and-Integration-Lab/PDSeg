#For execution format
import os
import shutil

import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader

from models import *
from functions import *
from utils import *
from segm.model.factory import create_segmenter
from segm import config
from datasets_wsss import Dataset_test_inf
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--val", action="store_true", default=False)
parser.add_argument("--test", action="store_true", default=True)
parser.add_argument("--test_on_train", action="store_true", default=False)
parser.add_argument("--ckpt_name", type=str, default=None)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--stage", type=int, default=1)

parser.add_argument("--model", type=str, default='segmenter')
parser.add_argument("--classes", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--dataset", type=str, default='BCSS-WSSS')

parser.add_argument("--weakly", action="store_true", default=False)
parser.add_argument("--affinity", action="store_true", default=False)
parser.add_argument("--mid_decode", action="store_true", default=True)
parser.add_argument("--cnn_distill", action="store_true", default=True)
parser.add_argument("--token_distill", action="store_true", default=True)
parser.add_argument("--path_work", default='weight/')
parser.add_argument("--consistency_loss", action="store_true", default=False)

args = parser.parse_args()


def get_model_config(args):
    backbone = 'vit_small_patch16_384'
    dataset = 'other'
    decoder = 'mask_transformer'
    dropout = 0.0
    drop_path = 0.1
    
    cfg = config.load_config()
    model_cfg = cfg["model"][backbone]
    dataset_cfg = cfg["dataset"][dataset]
    if "mask_transformer" in decoder:
        decoder_cfg = cfg["decoder"]["mask_transformer"]
    else:
        decoder_cfg = cfg["decoder"][decoder]

    im_size = 224
    crop_size = dataset_cfg.get("crop_size", im_size)

    model_cfg["image_size"] = (crop_size, crop_size)
    model_cfg["backbone"] = backbone
    model_cfg["dropout"] = dropout
    model_cfg["drop_path_rate"] = drop_path
    decoder_cfg["name"] = decoder
    model_cfg["decoder"] = decoder_cfg
    
    net_kwargs=model_cfg
    net_kwargs["n_cls"] = args.classes
    net_kwargs["mid_decode"] = args.mid_decode
    net_kwargs["cnn_distill"] = args.cnn_distill
    net_kwargs["distilled"] = args.token_distill
    
    return net_kwargs

class PDSeg():
    def __init__(self):
        super().__init__()
        net_kwargs = get_model_config(args)
        self.model = create_segmenter(net_kwargs)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        path_model = './weight/best_model.pth'
        self.model.load_state_dict(torch.load(path_model)['model_state'])
        self.model.eval()

    def init_parameters(self, image_path, preprocessed_result_path, result_file_name_list):
        self.image_path = image_path
        self.label2color = Label2Color(cmap=luad_cmap())
        dataset_size = [224, 224]
        dataset_test = Dataset_test_inf(dataset_size, self.device, self.image_path)
        self.dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4) 
        self.step = 0
        self.result_file_name_list = result_file_name_list
    
    def progress(self):
        progress = float(self.step/len(self.dataloader_test))*100
        return "%02d" % progress + '% '
    
    def execute(self):
        for image in (self.dataloader_test):
            self.step += 1
            image = image.to(self.device)
            with torch.no_grad():
                preds = self.model(torch.cat([image, image.flip(-1)], dim=0))
                if args.mid_decode:
                    fin = preds["fin_pix_masks"]
                else:
                    fin = preds["fin_pix_masks"]
                if args.stage == 1:
                    out = (fin[:1] + fin[1:].flip(-1)) / 2
            out = out.softmax(dim=1)
            pred = torch.argmax(out, dim=1).cpu().numpy().squeeze(0)
            pred_show = self.label2color(pred.astype(int))
            
            plt.figure()
            plt.axis('off')
            plt.imshow(pred_show)
            plt.savefig(self.result_file_name_list[self.step])
            

# For testing
AlgorithmName_runner = PDSeg()
image_path = './data/BCSS-WSSS/test/img/'
predict_result_path = './output'

if os.path.isdir(predict_result_path):
    shutil.rmtree(predict_result_path)
    os.makedirs(predict_result_path)
else:
    os.makedirs(predict_result_path)

preprocessed_result_path = ''
result_file_name_list = ['output/%04d.jpg'%i for i in range(len(os.listdir(image_path)))]

AlgorithmName_runner.init_parameters(image_path, preprocessed_result_path, result_file_name_list)

import threading
import time

def call_progress():
    while True:
        print('==========================progress: ' + str(AlgorithmName_runner.progress()) + '=====================================')
        
        time.sleep(10)

thread_progress = threading.Thread(target=call_progress)
thread_progress.start()
AlgorithmName_runner.execute()

