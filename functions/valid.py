import torch
import numpy as np
from sklearn import metrics
from utils import Evaluator
import torch.nn.functional as F

def valid(model, dataloader, test_num_pos, device, args):

    evaluator = Evaluator(args.classes)
    evaluator.reset()
    
    evaluator2 = Evaluator(args.classes)
    evaluator2.reset()

    with torch.no_grad():
        for image, target in dataloader:
            image, target = image.to(device), target.to(device)
            with torch.no_grad():
                preds = model(image)
            pred = torch.argmax((preds["fin_pix_masks"]) if args.model == 'segmenter' else preds[3], dim=1).cpu().numpy()
            target = target.squeeze(0).cpu().numpy()
            ## cls 4 is exclude
            pred[target==4]=4
            evaluator.add_batch(target, pred)
            
            if args.cnn_distill:
                pred2 = torch.argmax(preds["cnn_pix_masks"], dim=1).cpu().numpy()
                pred2[target==4]=4
                evaluator2.add_batch(target, pred2)

        if args.cnn_distill:
            Acc = evaluator2.Pixel_Accuracy()
            Acc_class = evaluator2.Pixel_Accuracy_Class()
            mIoU = evaluator2.Mean_Intersection_over_Union()
            ious = evaluator2.Intersection_over_Union()
            FWIoU = evaluator2.Frequency_Weighted_Intersection_over_Union()

            print('Middle Validation:')
            print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
            print('IoUs: ', ious)
        
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        ious = evaluator.Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        print('Final Validation:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('IoUs: ', ious)
        
        return mIoU