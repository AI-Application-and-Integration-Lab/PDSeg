import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from utils import hausdorff, gm, Evaluator, luad_cmap, Label2Color
import tqdm

def test(path_work, model, dataloader, device, args, weight=None):
    step = 0
    
    if weight is None:
        path_model = path_work + 'best_model.pth'
    elif weight == 'final':
        path_model = path_work + 'final_model.pth'
    else:
        path_model = path_work + weight

    print(f'Inference with weight: {path_model}')
    
    model.load_state_dict(torch.load(path_model)['model_state'])
    print(torch.load(path_model)['epoch'])
    model.eval()
    plt.ion()

    evaluator = Evaluator(args.classes)
    evaluator.reset()
    
    label2color = Label2Color(cmap=luad_cmap())

    with torch.no_grad():
        for image, target, other in tqdm.tqdm(dataloader):
            step += 1
            image, target = image.to(device), target.to(device)
            with torch.no_grad():
                preds = model(torch.cat([image, image.flip(-1)], dim=0))
                if args.model == 'segmenter':
                    if args.mid_decode:
                        # fin = (preds["fin_pix_masks"] + preds["mid_pix_mask"]) / 2
                        fin = preds["fin_pix_masks"]
                    else:
                        fin = preds["fin_pix_masks"]
                    if args.stage == 1:
                        out = (fin[:1] + fin[1:].flip(-1)) / 2
                    elif args.stage == 2:
                        out = fin[:1]
                else:
                    out = preds[3]
            
            out = out.softmax(dim=1)
            if args.test_on_train:
                out *= target[:, :, None, None]
            pred = torch.argmax(out, dim=1).cpu().numpy().squeeze(0)
            
            if args.test_on_train:
                file_name = other["file_name"][0].split('.')[0]
                np.save(f'./wsss_results/{args.dataset}/{file_name}', pred)
                continue
            
            target = target.squeeze(0).squeeze(0).cpu().numpy()
            ## cls 4 is exclude
            pred_show = label2color(pred.astype(int))
            target_show = label2color(target.astype(int))

            pred[target==4]=4
            evaluator.add_batch(target, pred)
            
            # print(image_show)
            # plt.figure()
            # plt.subplot(1, 3, 1)
            # plt.imshow(image_show.long().squeeze(0))
            # plt.title("%dth" % step)
            # plt.xticks([])
            # plt.yticks([])
            # plt.subplot(1, 3, 2)
            # plt.imshow(target_show)
            # plt.title("Ground truth")
            # plt.xticks([])
            # plt.yticks([])
            # plt.subplot(1, 3, 3)
            # plt.imshow(pred_show)
            # plt.title("Prediction")
            # plt.xticks([])
            # plt.yticks([])
            # # plt.pause(1)
            # plt.savefig("output/%04d.jpg" % step)

        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        ious = evaluator.Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

        print('Validation:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('IoUs: ', ious)
