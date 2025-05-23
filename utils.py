import torch
import cv2
import numpy as np
from skimage import measure
import torch.nn as nn
import torch.nn.functional as F

def gm(tensor, r):
    num = tensor.shape[0]
    output = torch.zeros(num, tensor.shape[1])
    for i in range(num):
        num_tensor = tensor[i].numel()
        tensor_r = tensor[i]**r
        sum_tensor = tensor_r.sum()
        average_tensor = sum_tensor/num_tensor
        pos_rate = (average_tensor+0.000001)**(1/r)
        output[i] = pos_rate
    return output

def ngwp_focal(outputs, focal=True, alpha=1e-5, lam=1e-2):
    bs, c, h, w = outputs.size()

    masks = F.softmax(outputs, dim=1)
    masks_ = masks.view(bs, c, -1)
    logits = outputs.view(bs, c, -1)
    # y_ngwp = (logits * (masks_ + alpha)).sum(-1) / (masks_+alpha).sum(-1)
    y_ngwp = (logits * masks_).sum(-1) / (1.0 + masks_.sum(-1))
    # y_ngwp = logits.max(-1)[0]

    # focal penalty loss
    if focal:
        y_focal = torch.pow(1 - masks_.mean(-1), 3) * torch.log(lam + masks_.mean(-1))
        y = y_ngwp + y_focal
    else:
        y = y_ngwp
    return y


def attention_cam(outputs, alpha=0.01):
    bs, c, h, w = outputs.size()

    masks = F.softmax(outputs, dim=1)
    masks_ = masks.view(bs, c, -1)
    logits = outputs.view(bs, c, -1)

    y_ngwp = (logits * (masks_ + alpha)).sum(-1) / (masks_+alpha).sum(-1)
    return y_ngwp


def bce_loss(outputs, labels, mode='ngwp', reduction='sum'):
    bs, c, h, w = outputs.size()
    if mode == 'ngwp':
        y = ngwp_focal(outputs)
    elif mode == 'att':
        y = attention_cam(outputs)
    else:
        logits = outputs.view(bs, c, -1)
        y = logits.mean(-1)

    bs, n_cls = labels.shape
    y = y[:, -n_cls:]

    if reduction == 'sum':
        l = F.binary_cross_entropy_with_logits(y, labels, reduction="none").sum(dim=1).mean()
    else:
        l = F.binary_cross_entropy_with_logits(y, labels)
    return l

def hausdorff(a, b):
    hausdorff_sd = cv2.createHausdorffDistanceExtractor()
    contours_a, hierarchy_a = cv2.findContours(a.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_b, hierarchy_b = cv2.findContours(b.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    temp1 = None
    for i in range(len(contours_a)):
        if i == 0:
            temp1 = contours_a[0]
        else:
            temp1 = np.concatenate((temp1, contours_a[i]), axis=0)
    if temp1 is not None:
        contours_a = temp1
    else:
        contours_a = np.zeros((1, 1, 2), dtype=int)

    temp2 = None
    for i in range(len(contours_b)):
        if i == 0:
            temp2 = contours_b[0]
        else:
            temp2 = np.concatenate((temp2, contours_b[i]), axis=0)
    if temp2 is not None:
        contours_b = temp2
    else:
        contours_b = np.zeros((1, 1, 2), dtype=int)

    hausdorff_distance = hausdorff_sd.computeDistance(contours_a, contours_b)
    return hausdorff_distance


def minmaxscaler(data):
    mindata = data.min()
    maxdata = data.max()
    return (data - mindata)/(maxdata-mindata)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def pseudo_gtmask(mask, ambiguous=True, cutoff_top=0.6, cutoff_bkg=0.6, cutoff_low=0.2, eps=1e-8, old_classes=16):
    """Convert continuous mask into binary mask"""
    bs, c, h, w = mask.size()
    mask = mask.view(bs, c, -1)

    # for each class extract the max confidence
    mask_max, _ = mask.max(-1, keepdim=True)
    # mask_max[:, :1] *= cutoff_bkg
    mask_max *= cutoff_top
    # mask_max *= cutoff_top

    # if the top score is too low, ignore it (at least cutoff_low)
    lowest = torch.Tensor([cutoff_low]).type_as(mask_max)
    mask_max = mask_max.max(lowest)

    pseudo_gt = (mask > mask_max).type_as(mask)

    # remove ambiguous pixels
    if ambiguous:
        ambiguous = (pseudo_gt.sum(1, keepdim=True) > 1).type_as(mask)
        pseudo_gt = (1 - ambiguous) * pseudo_gt

    return pseudo_gt.view(bs, c, h, w)

def balanced_mask_loss_ce(mask, pseudo_gt, gt_labels, ignore_index=255, old_classes=16):
    """Class-balanced CE loss
    - cancel loss if only one class in pseudo_gt
    - weight loss equally between classes
    """

    # b,c,h,w = pseudo_gt.shape
    mask = F.interpolate(mask, size=pseudo_gt.size()[-2:], mode="bilinear", align_corners=True)

    # indices of the max classes
    mask_gt = torch.argmax(pseudo_gt, 1)

    # for each pixel there should be at least one 1
    # otherwise, ignore
    ignore_mask = pseudo_gt.sum(1) < 1.
    mask_gt[ignore_mask] = ignore_index

    # class weight balances the loss w.r.t. number of pixels
    # because we are equally interested in all classes
    bs, c, h, w = pseudo_gt.size()
    num_pixels_per_class = pseudo_gt.view(bs, c, -1).sum(-1)  # BS, C -> pixel per class
    num_pixels_total = num_pixels_per_class.sum(-1, keepdim=True)  # BS -> pixel per image
    class_weight = (num_pixels_total - num_pixels_per_class) / (1 + num_pixels_total)  # BS, C
    class_weight = (pseudo_gt * class_weight[:, :, None, None]).sum(1).view(bs, -1)  # BS, H, W

    # BCE loss
    loss = F.cross_entropy(mask, mask_gt, ignore_index=ignore_index, reduction="none")
    loss = loss.view(bs, -1)

    # we will have the loss only for batch indices
    # which have all classes in pseudo mask
    gt_num_labels = gt_labels.sum(-1).type_as(loss)
    num_pixels_per_class *= gt_labels
    ps_num_labels = (num_pixels_per_class > 0).type_as(loss).sum(-1)
    batch_weight = (gt_num_labels == ps_num_labels).type_as(loss)

    loss = batch_weight * (class_weight * loss).mean(-1)
    return loss.mean()

def luad_cmap():
    cmap = np.zeros((256, 3), dtype=np.uint8)
    colors = [
        [205,51,51],
        [0,255,0],
        [65,105,225],
        [255,165,0],
        [255, 255, 255]
    ]

    for i in range(len(colors)):
        cmap[i] = colors[i]

    return cmap.astype(np.uint8)

class Label2Color(object):
    def __init__(self, cmap):
        self.cmap = cmap

    def __call__(self, lbls):
        return self.cmap[lbls]

# class ClassAwareTripletLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.triplet_loss = nn.TripletMarginLoss(margin=0.5, p=2)
#         self.pos_prot = None
#         self.neg_prot = None

#     def forward(self, inputs, targets, scores, label, only_update):
#         bs, ch, h, w = inputs.shape
#         tot_classes = scores.shape[1]
        
#         scores = F.interpolate(scores, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
#         # inputs = F.interpolate(inputs, size=scores.shape[-2:], mode="bilinear", align_corners=False)
#         # targets = F.interpolate(targets, size=scores.shape[-2:], mode="bilinear", align_corners=False)

#         # foreground
#         weight1 = scores[:, :1].detach().clone()
#         weight1 *= (weight1 > 0.5).float()
#         print((weight1.view(bs, -1)>0).sum(-1))
#         weight1 = weight1 / (weight1.view(bs, -1).sum(-1) + 1e-5)[:, None, None, None]
#         for_emb = (inputs * weight1).view(bs, ch, -1).sum(-1)
#         for_emb_norm = F.normalize(for_emb, p=2, dim=1)
        
#         loss = 0
#         if not only_update and label.sum() > 0:
#             # background
#             # weight2 = (1 - scores[:, :1]).detach().clone()
#             # weight2 *= (weight2 > 0.5).float()
#             # # print('bg', (weight2.view(bs, -1)>0).sum(-1))
#             # weight2 = weight2 / (weight2.view(bs, -1).sum(-1) + 1e-5)[:, None, None, None]
#             # bg_emb = (targets * weight2).view(bs, ch, -1).sum(-1)
#             # bg_emb_norm = F.normalize(bg_emb, p=2, dim=1)
            
#             anchors = for_emb_norm
#             positives = self.pos_prot.detach().clone()
#             negatives = self.neg_prot.detach().clone()
#             # negatives = bg_emb_norm.mean(dim=0, keepdim=True)
#             # print(anchors.shape, positives.shape, negatives.shape)
#             loss = []
#             for b in range(bs):
#                 if label[b]:
#                     loss.append(self.triplet_loss(anchors, positives, negatives))
            
#             loss = torch.stack(loss).mean()
        
#         if label.sum() > 0:
#             cur_prot = (label * for_emb_norm).sum(dim=0, keepdim=True) / label.sum()
#             if self.pos_prot is None:
#                 self.pos_prot = cur_prot
#             else:
#                 self.pos_prot = (0.5 * self.pos_prot) + (0.5 * cur_prot)
        
#         if (label == 0).any():
#             cur_prot = (1 - label) * targets.view(bs, ch, -1).sum(-1) / (1 - label).sum()
#             if self.neg_prot is None:
#                 self.neg_prot = cur_prot
#             else:
#                 self.neg_prot = (0.5 * self.neg_prot) + (0.5 * cur_prot)

#         return loss

class ClassAwareTripletLoss(nn.Module):
    def __init__(self, device, prot):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=0.2, p=2, reduction='none')
        self.device = device
        self.pos_prot = prot
        if self.pos_prot is not None:
            self.pos_prot = self.pos_prot.to(device)

    def forward(self, inputs, targets, scores, label, only_update):
        bs, ch, h, w = inputs.shape
        tot_classes = scores.shape[1]
        scores = F.interpolate(scores, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        """"""
        # scores = torch.softmax(scores, dim=1)
        """"""
        cur_embs_norm = []
        target_embs_norm = []
        for c in range(tot_classes):
            # weight = scores[:, c: c+1].detach().clone()
            # weight = weight / (weight.view(bs, -1).sum(-1) + 1e-5)[:, None, None, None]
            # cur_emb = (inputs * weight).view(bs, ch, -1).sum(-1)
            # cur_emb_norm = F.normalize(cur_emb, p=2, dim=1)
            # cur_embs_norm.append(cur_emb_norm)
            
            # max
            score_c = scores[:, c: c+1].view(bs, 1, -1)
            max_score = score_c.max(dim=-1, keepdims=True)[0]
            weight = (score_c == max_score).float()
            target_emb_norm = (inputs.view(bs, ch, -1) * weight).sum(-1)
            target_emb_norm = F.normalize(target_emb_norm, p=2, dim=1)
            target_embs_norm.append(target_emb_norm)
            
            cur_embs_norm.append(target_emb_norm)
        
        # bs x cls x channels
        cur_embs_norm = torch.stack(cur_embs_norm).permute(1, 0, 2)
        target_embs_norm = torch.stack(target_embs_norm).permute(1, 0, 2)
        
        loss = 0
        if not only_update and label.sum() > 0:
            # dist = torch.cdist(cur_embs_norm, cur_embs_norm) # bs x cls x cls
            # dist = torch.cdist(cur_embs_norm, target_embs_norm) # bs x cls x cls
            prot_dist = torch.cdist(cur_embs_norm, self.pos_prot.repeat(bs, 1, 1))
            
            # mask: 1 means ignore, bs x tot_classes x tot_classes
            mask = (1 - label[:, None, :, 0]).repeat(1, tot_classes, 1)
            diagonal = torch.eye(tot_classes, device=self.device, dtype=torch.bool)[None, :, :]
            # assign ignore position with maximum similariry: 1.0
            mask.masked_fill_(diagonal, 1.0)
            
            loss = []
            for c in range(tot_classes):
                anchors = cur_embs_norm[:, c]
                positives = self.pos_prot[c].repeat(bs, 1)

                # sort_pos = torch.argsort(dist, dim=2)
                sort_prot_pos = torch.argsort(prot_dist, dim=2)
                negatives = []
                for b in range(bs):
                    # if (1 - sort_pos[b, c]).sum() > 1:
                    #     # print(1)
                    #     for pos in sort_pos[b, c]:
                    #         if mask[b, c, pos] == 0:
                    #             neg_cls = pos
                    #             break
                    #     negatives.append(cur_embs_norm[b, neg_cls])
                    # else:
                        # print(2)
                    for pos in sort_prot_pos[b, c]:
                        if pos == c:
                            continue
                        neg_cls = pos
                        break
                    negatives.append(self.pos_prot[neg_cls])
                    
                    # print(c, neg_cls)
                negatives = torch.stack(negatives)
                loss.append(self.triplet_loss(anchors, positives.detach(), negatives.detach()))
            
            loss = torch.stack(loss).transpose(0, 1)
            loss = ((loss * label[:, :, 0]).sum(-1) / label[:, :, 0].sum(-1)).mean()
        
        if label.sum() > 0:
            label_ = label.sum(dim=0)
            cur_prot = (label * target_embs_norm).sum(dim=0) / (label_ + 1e-5)
            
            if self.pos_prot is None:
                self.pos_prot = cur_prot.detach()
            else:
                for c, l in enumerate(label_):
                    if l > 0:
                        if (self.pos_prot[c] == 0).all():
                            self.pos_prot[c] = cur_prot[c].detach()
                        else:
                            self.pos_prot[c] = (0.5 * self.pos_prot[c]) + (0.5 * cur_prot[c].detach())
                            
        return loss
    
    
class ClassAwareTripletLoss(nn.Module):
    def __init__(self, device, prot):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=0.2, p=2, reduction='none')
        self.device = device
        self.pos_prot = prot
        if self.pos_prot is not None:
            self.pos_prot = self.pos_prot.to(device)

    def forward(self, inputs, label, only_update):
        bs, tot_classes, _ = inputs.shape
        
        cur_embs_norm = []
        target_embs_norm = []
        for c in range(tot_classes):
            target_emb_norm = inputs[:, c]
            target_emb_norm = F.normalize(target_emb_norm, p=2, dim=1)
            target_embs_norm.append(target_emb_norm)
            
            cur_embs_norm.append(target_emb_norm)
        
        # bs x cls x channels
        cur_embs_norm = torch.stack(cur_embs_norm).permute(1, 0, 2)
        target_embs_norm = torch.stack(target_embs_norm).permute(1, 0, 2)
        
        loss = 0
        if not only_update and label.sum() > 0:
            prot_dist = torch.cdist(cur_embs_norm, self.pos_prot.repeat(bs, 1, 1))
            
            # mask: 1 means ignore, bs x tot_classes x tot_classes
            mask = (1 - label[:, None, :, 0]).repeat(1, tot_classes, 1)
            diagonal = torch.eye(tot_classes, device=self.device, dtype=torch.bool)[None, :, :]
            # assign ignore position with maximum similariry: 1.0
            mask.masked_fill_(diagonal, 1.0)
            
            loss = []
            for c in range(tot_classes):
                anchors = cur_embs_norm[:, c]
                positives = self.pos_prot[c].repeat(bs, 1)

                # sort_pos = torch.argsort(dist, dim=2)
                sort_prot_pos = torch.argsort(prot_dist, dim=2)
                negatives = []
                for b in range(bs):
                    for pos in sort_prot_pos[b, c]:
                        if pos == c:
                            continue
                        neg_cls = pos
                        break
                    negatives.append(self.pos_prot[neg_cls])
                    
                    # print(c, neg_cls)
                negatives = torch.stack(negatives)
                loss.append(self.triplet_loss(anchors, positives.detach(), negatives.detach()))
            
            loss = torch.stack(loss).transpose(0, 1)
            loss = ((loss * label[:, :, 0]).sum(-1) / label[:, :, 0].sum(-1)).mean()
        
        if label.sum() > 0:
            label_ = label.sum(dim=0)
            cur_prot = (label * target_embs_norm).sum(dim=0) / (label_ + 1e-5)
            
            if self.pos_prot is None:
                self.pos_prot = cur_prot.detach()
            else:
                for c, l in enumerate(label_):
                    if l > 0:
                        if (self.pos_prot[c] == 0).all():
                            self.pos_prot[c] = cur_prot[c].detach()
                        else:
                            self.pos_prot[c] = (0.9 * self.pos_prot[c]) + (0.1 * cur_prot[c].detach())
                            
        return loss
 
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        ious = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(ious)
        return MIoU

    def Intersection_over_Union(self):
        ious = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return ious

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image<self.num_class)
        label = (self.num_class) * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=(self.num_class)**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        
        
class BCEWithLogitsLossWithWeights(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, targets, weight):
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        if self.reduction == 'mean':
            # if targets have only zeros, we skip them
            return torch.masked_select(loss, weight.sum(dim=1) != 0).mean()
        elif self.reduction == 'sum':
            return torch.masked_select(loss, weight.sum(dim=1) != 0).sum()
        else:
            return loss * targets.sum(dim=1)
        
class BCEWithLogitsLossWithIgnoreIndex(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        # targets is B x C x H x W so shape[1] is C
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        if self.reduction == 'mean':
            # if targets have only zeros, we skip them
            return torch.masked_select(loss, targets.sum(dim=1) != 0).mean()
        elif self.reduction == 'sum':
            return torch.masked_select(loss, targets.sum(dim=1) != 0).sum()
        else:
            return loss * targets.sum(dim=1)