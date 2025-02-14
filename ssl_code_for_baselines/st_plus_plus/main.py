from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils import count_params, meanIOU, color_map

from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import select_reliable,init_basic_elems
from trainer import train, label
from parameters import parse_args
MODE = None




def main(args):
    NUM_WORKERS = 0
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)
    if args.plus and args.reliable_id_path is None:
        exit('Please specify reliable-id-path in ST++.')

    criterion = CrossEntropyLoss(ignore_index=255)

    valset = SemiDataset(mode='val',size=None, labeled_id_path=args.val_path)
    valloader = DataLoader(valset, batch_size=1,shuffle=False, pin_memory=True, num_workers=NUM_WORKERS, drop_last=False)

    # <====================== Supervised training with labeled images (SupOnly) ======================>
    print('\n================> Total stage 1/%i: '
          'Supervised training on labeled images (SupOnly)' % (6 if args.plus else 3))

    global MODE
    MODE = 'train'
    SAVE_METRIC = 'Dice'

    trainset = SemiDataset( mode=MODE, size=args.crop_size, labeled_id_path=args.labeled_id_path)

    trainset.ids = 2 * trainset.ids if len(trainset.ids) < 200 else trainset.ids

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,pin_memory=True, num_workers=NUM_WORKERS, drop_last=True)

    pretrained_model_path = None 
    model, optimizer = init_basic_elems(args, pretrained_path=pretrained_model_path)

    print('\nParams: %.1fM' % count_params(model))
    # if not pretrained_model_path:

    # best_model, checkpoints = train(model, trainloader, valloader, criterion, optimizer, args)
    best_model, checkpoints = train(model, trainloader, valloader, criterion, optimizer, args, save_best_by=SAVE_METRIC)

    """
        ST framework without selective re-training
    """
    if not args.plus:
        # <============================= Pseudo label all unlabeled images =============================>
        print('\n\n\n================> Total stage 2/3: Pseudo labeling all unlabeled images')

        dataset = SemiDataset('label', None, None, args.unlabeled_id_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

        label(best_model, dataloader, args, save_best_by=SAVE_METRIC)

        # <======================== Re-training on labeled and unlabeled images ========================>
        print('\n\n\n================> Total stage 3/3: Re-training on labeled and unlabeled images')

        MODE = 'semi_train'

        trainset = SemiDataset(MODE, args.crop_size, args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=16, drop_last=True)

        model, optimizer = init_basic_elems(args)

        train(model, trainloader, valloader, criterion, optimizer, args, save_best_by=SAVE_METRIC)

        return

    """
        ST++ framework with selective re-training
    """
    # <===================================== Select Reliable IDs =====================================>
    print('\n\n\n================> Total stage 2/6: Select reliable images for the 1st stage re-training')

    dataset = SemiDataset('label', None, None, args.unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS, drop_last=False)

    select_reliable(checkpoints, dataloader, args, reliability_metric=SAVE_METRIC)

    # <================================ Pseudo label reliable images =================================>
    print('\n\n\n================> Total stage 3/6: Pseudo labeling reliable images')

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
    dataset = SemiDataset( 'label', None, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS, drop_last=False)

    label(best_model, dataloader, args,save_best_by=SAVE_METRIC)

    # <================================== The 1st stage re-training ==================================>
    print('\n\n\n================> Total stage 4/6: The 1st stage re-training on labeled and reliable unlabeled images')

    MODE = 'semi_train'

    trainset = SemiDataset(MODE, args.crop_size, args.labeled_id_path, cur_unlabeled_id_path, args.pseudo_mask_path)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS, drop_last=True)

    model, optimizer = init_basic_elems(args)

    best_model = train(model, trainloader, valloader, criterion, optimizer, args, save_best_by=SAVE_METRIC)

    # <=============================== Pseudo label unreliable images ================================>
    print('\n\n\n================> Total stage 5/6: Pseudo labeling unreliable images')

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'unreliable_ids.txt')
    dataset = SemiDataset( 'label', None, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    label(best_model, dataloader, args,save_best_by=SAVE_METRIC)

    # <================================== The 2nd stage re-training ==================================>
    print('\n\n\n================> Total stage 6/6: The 2nd stage re-training on labeled and all unlabeled images')

    trainset = SemiDataset( MODE, args.crop_size,
                           args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    model, optimizer = init_basic_elems(args)

    train(model, trainloader, valloader, criterion, optimizer, args, save_best_by=SAVE_METRIC)




if __name__ == '__main__':
    args = parse_args()

    # if args.epochs is None:
    #     args.epochs = {'pascal': 80, 'cityscapes': 240}[args.dataset]
    # if args.lr is None:
    #     args.lr = {'pascal': 0.001, 'cityscapes': 0.004}[args.dataset] / 16 * args.batch_size
    # if args.crop_size is None:
    #     args.crop_size = {'pascal': 321, 'cityscapes': 721}[args.dataset]

    print()
    print(args)

    main(args)







# def train(model, trainloader, valloader, criterion, optimizer, args):
#     iters = 0
#     total_iters = len(trainloader) * args.epochs

#     previous_best = 0.0

#     global MODE

#     if MODE == 'train':
#         checkpoints = []

#     for epoch in range(args.epochs):
#         print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
#               (epoch, optimizer.param_groups[0]["lr"], previous_best))

#         model.train()
#         total_loss = 0.0
#         tbar = tqdm(trainloader)

#         for i, (img, mask) in enumerate(tbar):
#             img, mask = img.cuda(), mask.cuda()

#             pred = model(img)
#             loss = criterion(pred, mask)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#             iters += 1
#             lr = args.lr * (1 - iters / total_iters) ** 0.9
#             optimizer.param_groups[0]["lr"] = lr
#             optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0

#             tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))

#         metric = meanIOU(num_classes=6)

#         model.eval()
#         tbar = tqdm(valloader)

#         with torch.no_grad():
#             for img, mask, _ in tbar:
#                 img = img.cuda()
#                 pred = model(img)
#                 pred = torch.argmax(pred, dim=1)

#                 metric.add_batch(pred.cpu().numpy(), mask.numpy())
#                 mIOU = metric.evaluate()[-1]

#                 tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

#         mIOU *= 100.0
#         if mIOU > previous_best:
#             if previous_best != 0:
#                 os.remove(os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
#             previous_best = mIOU
#             torch.save(model.module.state_dict(),
#                        os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))

#             best_model = deepcopy(model)

#         if MODE == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
#             checkpoints.append(deepcopy(model))

#     if MODE == 'train':
#         return best_model, checkpoints

#     return best_model
