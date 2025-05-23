import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
import os
import argparse

from models import *
from functions import *
from utils import *
from segm.model.factory import create_segmenter
from segm import config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--test_on_train", action="store_true", default=False)
    parser.add_argument("--ckpt_name", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--stage", type=int, default=1)
    
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--classes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dataset", type=str, default=None)
    
    parser.add_argument("--weakly", action="store_true", default=False)
    parser.add_argument("--affinity", action="store_true", default=False)
    parser.add_argument("--mid_decode", action="store_true", default=False)
    parser.add_argument("--cnn_distill", action="store_true", default=False)
    parser.add_argument("--token_distill", action="store_true", default=False)
    parser.add_argument("--path_work", default='checkpoints/')
    parser.add_argument("--consistency_loss", action="store_true", default=False)

    args = parser.parse_args()
    print(args)
    
    if args.dataset == 'glas':
        from datasets_glas import Dataset_train, Dataset_valid, Dataset_test
    else:
        from datasets_wsss import Dataset_train, Dataset_valid, Dataset_test
    
    print('Loading......')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset_name = args.dataset

    path_work = args.path_work
    if os.path.exists(path_work) is False:
        os.mkdir(path_work)

    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset_size = [224, 224]
    dataset_train = Dataset_train(dataset_size, device, dataset_name, args.test_on_train, args.stage)
    dataset_valid = Dataset_valid(dataset_size, device, dataset_name)
    dataset_test = Dataset_test(dataset_size, device, dataset_name)

    batch_size = args.batch_size
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size if not args.test_on_train else 1, shuffle=True, num_workers=4, drop_last=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False, num_workers=4)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)
    test_num_pos = 80

    if args.model == 'segmenter':
        net_kwargs = get_model_config(args)
        model = create_segmenter(net_kwargs)
        model.to(device)
        hyperparameters = {
            'r' : 4,
            'lr' : 0.01, # 0.01
            'wd' : 0.0,
            'start_epoch': 0,
            'epoch' : 40,
            'pretrain' : False,
            'optimizer' : 'sgd', 
            'prot': None
        }
    else:
        model = Swin_MIL(classes=args.classes).to(device)
        hyperparameters = {
            'r' : 4,
            'lr' : 1e-6,
            'wd' : 0.0005,
            'start_epoch': 0,
            'epoch' : 50,
            'pretrain' : True,
            'optimizer' : 'side',  # side
            'prot': None
        }
    
    ckpt_name = 'final_model.pth'
    if args.ckpt_name:
        ckpt_name = f'final_model_{args.ckpt_name}.pth'
    
    if args.resume:
        checkpoint = torch.load(os.path.join(path_work, ckpt_name), map_location="cpu")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        hyperparameters['start_epoch'] = epoch = checkpoint['epoch']
        hyperparameters['pretrain'] = False
        print(f'resume from {os.path.join(path_work, ckpt_name)}, epoch: {epoch}')

    print('Dataset: ' + dataset_name)
    print('Data Volume: ', len(dataloader_train.dataset))
    print('Model: ', type(model))
    print('Total Classes: ', args.classes)
    print('Batch Size: ', batch_size)
    
    if not args.test:
        train(path_work, model, dataloader_train, device, hyperparameters, valid if args.val else None, dataloader_valid, test_num_pos, args)
    

    # 使用 best model 來 inference
    ckpt_name = 'best_model.pth'
    if args.ckpt_name:
        ckpt_name = f'best_model_{args.ckpt_name}.pth'

    test(path_work, model, dataloader_test if not args.test_on_train else dataloader_train, device, args, ckpt_name)

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

    # model config
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

if __name__ == '__main__':
    main()