
import torch
import os
from copy import deepcopy
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from utils import count_params, meanIOU, color_map
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from tqdm import tqdm
from utils import *



def train(model, trainloader, valloader, criterion, optimizer, args, save_best_by="mIoU"):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {DEVICE}")

    iters = 0
    total_iters = len(trainloader) * args.epochs
    best_mIoU, best_dice = 0.0, 0.0  # Track best scores

    global MODE
    if MODE == 'train':
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
                os.remove(os.path.join(args.save_path, f"{args.model}_{args.backbone}_{best_metric:.2f}.pth"))

            if save_best_by == "mIoU":
                best_mIoU = mIOU
            else:
                best_dice = dice_avg

            model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(model_state, os.path.join(args.save_path, f"{args.model}_{args.backbone}_{save_metric:.2f}.pth"))
            best_model = deepcopy(model)

        # Save Checkpoints at Certain Epochs
        if MODE == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
            checkpoint_path = os.path.join(args.save_path, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

            checkpoints.append(deepcopy(model))

    return best_model, checkpoints if MODE == 'train' else best_model




def label(model, dataloader, args, save_best_by="mIoU"):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    model.eval()
    tbar = tqdm(dataloader)

    metric = meanIOU(num_classes=6)
    cmap = color_map()

    dice_total = 0.0
    num_batches = 0

    with torch.no_grad():
        for img, mask, id in tbar:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            pred = model(img)["out"]  # Extract "out" from DeepLabV3 output
            pred = torch.argmax(pred, dim=1).cpu()

            metric.add_batch(pred.numpy(), mask.cpu().numpy())  # Ensure masks are on CPU
            mIOU = metric.evaluate()[-1]
            dice = dice_score(pred, mask.cpu(), num_classes=6, exclude_background=True)

            dice_total += dice
            num_batches += 1

            pred_img = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred_img.putpalette(cmap)

            # Save the pseudo-mask
            pred_img.save(os.path.join(args.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))

            # Show progress with chosen metric
            if save_best_by == "mIoU":
                tbar.set_description(f'mIoU: {mIOU * 100:.2f}')
            else:
                tbar.set_description(f'Dice: {dice:.4f}')

    avg_mIoU = metric.evaluate()[-1] * 100.0
    avg_dice = dice_total / num_batches

    print(f"\nFinal Pseudo-Labeling Results: mIoU = {avg_mIoU:.2f}, Dice = {avg_dice:.4f}")

    return avg_mIoU if save_best_by == "mIOU" else avg_dice
