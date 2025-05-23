import torch
import torch.nn as nn
import math
from utils import gm, rand_bbox, bce_loss, pseudo_gtmask, balanced_mask_loss_ce, ngwp_focal,  ClassAwareTripletLoss, BCEWithLogitsLossWithWeights, BCEWithLogitsLossWithIgnoreIndex
from wss import *
import torch.nn.functional as F
import torchvision
import numpy as np
import argparse
from segm.optim.factory import create_optimizer, create_scheduler
import os

def train(path_work, model, dataloader_train, device, hp, valid_fn=None, dataloader_valid=None, test_num_pos=0, args=None):
    
    r = hp['r']
    lr = hp['lr']
    wd = hp['wd']
    num_epoch = hp['epoch']
    start_epoch = hp['start_epoch']
    best_result = 0
    print('Learning Rate: ', lr)
    loss_fn = nn.BCELoss()
    criterion = BCEWithLogitsLossWithIgnoreIndex(reduction='none')
    dataset_size = len(dataloader_train.dataset)

    if hp['optimizer'] == 'side':
        params1 = list(map(id, model.decoder1.parameters()))
        params2 = list(map(id, model.decoder2.parameters()))
        params3 = list(map(id, model.decoder3.parameters()))
        # params4 = list(map(id, model.linear_fuse.parameters()))
        base_params = filter(lambda p: id(p) not in params1 + params2 + params3, model.parameters())
        params = [{'params': base_params},
                  {'params': model.decoder1.parameters(), 'lr': lr / 100, 'weight_decay': wd},
                  {'params': model.decoder2.parameters(), 'lr': lr / 100, 'weight_decay': wd},
                  {'params': model.decoder3.parameters(), 'lr': lr / 100, 'weight_decay': wd},
                #   {'params': model.linear_fuse.parameters(), 'lr': lr / 100, 'weight_decay': wd}
                  ]
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    
    elif hp['optimizer'] == 'sgd':
        # optimizer
        optimizer_kwargs = dict(
            opt='sgd',
            lr=lr,
            weight_decay=0.0,
            momentum=0.9,
            clip_grad=None,
            sched='polynomial',
            epochs=num_epoch,
            min_lr=1e-5,
            poly_power=0.9,
            poly_step_size=1,
        )
        print(optimizer_kwargs)
        optimizer_kwargs["iter_max"] = len(dataloader_train) * optimizer_kwargs["epochs"]
        optimizer_kwargs["iter_warmup"] = 0.0
        opt_args = argparse.Namespace()
        opt_vars = vars(opt_args)
        for k, v in optimizer_kwargs.items():
            opt_vars[k] = v
        optimizer = create_optimizer(opt_args, model)
        lr_scheduler = create_scheduler(opt_args, optimizer)
    
    print("{:*^50}".format("training start"))
    for epoch in range(start_epoch, num_epoch):
        model.train()
        epoch_loss = 0
        step = 0
        batch_num = len(dataloader_train)
        num_updates = epoch * len(dataloader_train)

        for index, batch in enumerate(dataloader_train):
            image, label, other = batch
        
            img_show = other["img_show"]
            image = image.to(device)
            label = label.to(device)
            img_show = img_show.to(device)
            bs = image.shape[0]
            
            if args.cnn_distill:
                model.eval()
                with torch.no_grad():
                    if args.model == 'segmenter':
                        out = model(image)
                        pred = out["cnn_patch_masks"].detach()
                        pix_pred = out["fin_pix_masks"].detach()
            
            model.train()
            if args.model == 'segmenter':
                output = model(image)
                
                if args.mid_decode:
                    mid_masks = output["mid_patch_mask"]
                    
                if not args.weakly: # 如果不是弱監督的話
                    fin_masks = output["fin_pix_masks"]
                    pix_label = label.long()
                    loss = criterion(fin_masks, pix_label).mean()
                
                else:
                    fin_masks = output["fin_patch_masks"]
                    y = ngwp_focal(fin_masks)

                    lossf = loss_fn(torch.sigmoid(y), label)
                    
                    if args.cnn_distill:
                        cnn_masks = output["cnn_patch_masks"]
                        y_cnn = ngwp_focal(cnn_masks)
                        lossf += loss_fn(torch.sigmoid(y_cnn), label)
                        
                        if epoch >= 1:
                            alpha = 0.5
                            int_masks_orig = pred.softmax(dim=1)
                            pseudo_gt_seg_lx = binarize(int_masks_orig)
                            pseudo_gt_seg_lx = (alpha * pseudo_gt_seg_lx) + \
                                            ((1-alpha) * int_masks_orig)
                            lossf += F.binary_cross_entropy_with_logits(mid_masks, pseudo_gt_seg_lx)
                    
                    loss = lossf
            
            else:
                side1, side2, side3, fusion, feat = model(image)
                loss1 = loss_fn(gm(side1, r).to(device), label)
                loss2 = loss_fn(gm(side2, r).to(device), label)
                loss3 = loss_fn(gm(side3, r).to(device), label)
                lossf = loss_fn(gm(fusion, r).to(device), label)
                
                loss = loss1 + loss2 + loss3 + lossf
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.model == 'segmenter':
                num_updates += 1
                lr_scheduler.step_update(num_updates=num_updates)

            epoch_loss += loss.item()
            step += 1
            
            if index % 100 == 0:
                print("batch %d/%d loss:%0.4f" % (index, batch_num, loss.item()))
        
        epochs = epoch + 1
        average_loss = epoch_loss / math.ceil(dataset_size // dataloader_train.batch_size)
        print("epoch %d loss:%0.4f" % (epochs, average_loss))

        state = {"model_state": model.state_dict(), 
                "epoch": epochs}
        
        if valid_fn is not None:
            model.eval()
            result = valid_fn(model, dataloader_valid, test_num_pos, device, args)
            print('epoch %d loss:%.4f result:%.3f' % (epochs, average_loss, result))
            print(result, best_result)
            if result > best_result:
                best_result = result
                ckpt_name = 'best_model.pth'
                if args.ckpt_name:
                    ckpt_name = f'best_model_{args.ckpt_name}.pth'
                torch.save(state, path_work + ckpt_name)
        
        ckpt_name = 'final_model.pth'
        if args.ckpt_name:
            ckpt_name = f'final_model_{args.ckpt_name}.pth'
        
    torch.save(state, os.path.join(path_work, ckpt_name))
    
    print('best result: %.3f' % best_result)

def binarize(input):
    max = input.max(dim=1, keepdim=True)[0]
    return (input >= max).type_as(input)