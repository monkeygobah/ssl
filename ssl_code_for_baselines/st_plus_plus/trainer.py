
from copy import deepcopy
from torch.optim.lr_scheduler import LambdaLR
from utils import meanIOU, color_map
import numpy as np
import os
from PIL import Image
import torch
from tqdm import tqdm
from utils import *


def train(model, trainloader, valloader, criterion, optimizer, args, save_best_by="mIoU", mode= 'train'):
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {DEVICE}")

    iters = 0
    total_iters = len(trainloader) * args.epochs
    best_mIoU, best_dice = 0.0, 0.0  # Track best scores

    # global mode
    if mode == 'train':
        checkpoints = []

    # Learning Rate Scheduler (PolyLR)
    def poly_lr_lambda(epoch):
        return (1 - epoch / args.epochs) ** 0.9  # Standard poly LR decay

    scheduler = LambdaLR(optimizer, lr_lambda=poly_lr_lambda)

    for epoch in range(args.epochs):
        print(f"\n==> Epoch {epoch + 1}/{args.epochs}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        model.train()
        total_loss = 0.0
        tbar = tqdm(trainloader)

        for i, (img, mask) in enumerate(tbar):

            img, mask = img.to(DEVICE), mask.to(DEVICE)
 
            pred = model(img)["out"]  
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            iters += 1

            tbar.set_description(f'Loss: {total_loss / (i + 1):.4f}')

        scheduler.step()  

        # Validation
        metric = meanIOU(num_classes=6)
        dice_total = 0.0
        num_batches = 0

        model.eval()
        tbar = tqdm(valloader)

        with torch.no_grad():
            for img, mask, _ in tbar:
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                pred = model(img)["out"]
                pred = torch.argmax(pred, dim=1)

                metric.add_batch(pred.cpu().numpy(), mask.cpu().numpy())  
                dice_total += dice_score(pred, mask, num_classes=6, exclude_background=True)  
                num_batches += 1

                tbar.set_description(f'mIoU: {metric.evaluate()[-1] * 100:.2f}, Dice: {dice_total / num_batches:.4f}')

        mIOU = metric.evaluate()[-1] * 100.0
        dice_avg = dice_total / num_batches

        # Save Best Model (mIoU or Dice)
        save_metric = mIOU if save_best_by == "mIoU" else dice_avg
        best_metric = best_mIoU if save_best_by == "mIoU" else best_dice

        if save_metric > best_metric:
            if best_metric != 0:
                os.remove(os.path.join(args.save_path, f"dlv3_resnet101_{best_metric:.2f}.pth"))

            if save_best_by == "mIoU":
                best_mIoU = mIOU
            else:
                best_dice = dice_avg

            model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(model_state, os.path.join(args.save_path, f"dlv3_resnet101_{save_metric:.2f}.pth"))
            best_model = deepcopy(model)

        # Save Checkpoints at Certain Epochs
        if mode == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
            checkpoint_path = os.path.join(args.save_path, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

            checkpoints.append(deepcopy(model))

    return best_model, checkpoints if mode == 'train' else best_model





def label(model, dataloader, args, save_best_by="mIoU"):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Labeling pseudo-masks on: {DEVICE}")

    model.to(DEVICE)
    model.eval()
    tbar = tqdm(dataloader)

    cmap = color_map()  
    with torch.no_grad():
        for img, id in tbar:
            img = img.to(DEVICE)
            pred = model(img)["out"]
            pred = torch.argmax(pred, dim=1).cpu()

            # Convert prediction to an image
            pred_img = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred_img.putpalette(cmap)

            # Extract the original image filename and replace extension with ".png" with cmap
            # original_filename = os.path.basename(id[0])
            # pseudo_mask_filename = os.path.splitext(original_filename)[0] + ".png"
            # save_path = os.path.join(args.pseudo_mask_path, pseudo_mask_filename)
            # pred_img.save(save_path)

            # trying to save as pxels
            # Extract the original image filename and replace the extension with ".png"
            original_filename = os.path.basename(id[0])
            pseudo_mask_filename = os.path.splitext(original_filename)[0] + ".png"

            # Save only the grayscale pseudomask
            save_path = os.path.join(args.pseudo_mask_path, pseudo_mask_filename)
            pred_img.save(save_path)

            tbar.set_description(f"Saved pseudo-mask: {pseudo_mask_filename}")

    print(f"Pseudo-labeling complete. Masks saved in: {args.pseudo_mask_path}")


def load_pretrained_models(args, model, optimizer):

    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Paths to the best model and checkpoints
    best_model_pth = '/root/ssl/ssl/ssl_code_for_baselines/st_plus_plus/st_plus_testing/original_training/dlv3_resnet101_0.81.pth'
    checkpoint_paths = [
        '/root/ssl/ssl/ssl_code_for_baselines/st_plus_plus/st_plus_testing/original_training/checkpoint_epoch_6.pth',
        '/root/ssl/ssl/ssl_code_for_baselines/st_plus_plus/st_plus_testing/original_training/checkpoint_epoch_13.pth',
        '/root/ssl/ssl/ssl_code_for_baselines/st_plus_plus/st_plus_testing/original_training/checkpoint_epoch_20.pth'
    ]

    # Load best model
    print(f"Loading best model from: {best_model_pth}")
    model.load_state_dict(torch.load(best_model_pth, map_location=DEVICE))
    best_model = deepcopy(model).to(DEVICE)

    # Load checkpoints
    checkpoints = []
    for ckpt_pth in checkpoint_paths:
        if os.path.exists(ckpt_pth):
            print(f"Loading checkpoint from: {ckpt_pth}")
            checkpoint = torch.load(ckpt_pth, map_location=DEVICE)

            # Restore model weights
            model.load_state_dict(checkpoint['model_state_dict'])
            checkpoint_model = deepcopy(model).to(DEVICE)

            # Restore optimizer state (optional, needed if you want to resume training)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            checkpoints.append(checkpoint_model)
        else:
            print(f"Warning: Checkpoint file {ckpt_pth} not found!")

    print("Pretrained model and checkpoints loaded successfully.")
    return best_model, checkpoints
