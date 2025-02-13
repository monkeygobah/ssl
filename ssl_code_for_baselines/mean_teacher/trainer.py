import time
import torch
from mt_utils import *
from ramp_up import get_current_consistency_weight
from tqdm import tqdm
import matplotlib.pyplot as plt


# Check if the GPU is available
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {DEVICE}')

NO_LABEL = -1
CUDA_LAUNCH_BLOCKING=1
global_step = 0


def train_one_epoch(train_loader, 
          model, 
          ema_model, 
          optimizer, 
          epoch, 
          batch_size, 
          alpha,
          total_epochs=25,
          visualize=False,
          cons_style='linear',
          consistency_rampup = 5,
          consistency = 10,
          sweep = False,
          thresh = 3):

    global global_step

    consistency_criterion = softmax_mse_loss

    model.train()
    ema_model.train()
 
    epoch_student_loss = 0.0
    epoch_class_loss = 0.0
    epoch_consistency_loss = 0.0

    epoch_teacher_loss = 0.0  
    epoch_student_dice_per_class = [0.0]*6
    epoch_teacher_dice_per_class = [0.0]*6

    epoch_start = time.time()

    for i, (images, ema_images, labels, true_labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}", unit="batch") ):
        images = images.to(DEVICE)
        ema_images = ema_images.to(DEVICE)
        labels = labels.to(DEVICE)      
        true_labels = true_labels.to(DEVICE)  

        student_out = model(images)["out"]
        with torch.no_grad():
            teacher_out = ema_model(ema_images)["out"]  

        preds = student_out.argmax(dim=1)

        valid_mask = (labels != NO_LABEL)  
        labels_squeezed = labels.squeeze(1).clone()

        student_sup_loss = cross_entropy2d(student_out,labels_squeezed)
        teacher_sup_loss = cross_entropy2d(teacher_out,labels_squeezed)


        consistency_weight = get_current_consistency_weight(epoch,style=cons_style,consistency = consistency, consistency_rampup = consistency_rampup, thresh = thresh)
        con_loss = consistency_weight * consistency_criterion(student_out, teacher_out)

        loss = student_sup_loss + con_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        global_step += 1
        update_ema(model, ema_model, alpha, global_step)

        epoch_student_loss += loss.item()
        epoch_class_loss += student_sup_loss.item()
        epoch_consistency_loss += con_loss.item()
        epoch_teacher_loss += teacher_sup_loss.item()


        # valid_mask: shape [B, H, W]
        valid_mask_flat = valid_mask.squeeze(1).reshape(-1)  # shape [B*H*W]
        student_out_flat = student_out.permute(0, 2, 3, 1).reshape(-1, student_out.shape[1])  # [B*H*W, 6]
        true_labels_flat = true_labels.squeeze(1).reshape(-1)  # [B*H*W]
        
        student_out_valid = student_out_flat[valid_mask_flat]
        true_labels_valid = true_labels_flat[valid_mask_flat]
       
        if visualize:
            visualize_predictions(images, preds, labels, max_samples=2)

        epoch_student_dice_per_class, epoch_teacher_dice_per_class = get_dice_scores(student_out_valid, teacher_out,true_labels_valid,valid_mask_flat,epoch_student_dice_per_class,epoch_teacher_dice_per_class)

        if visualize:
            vis_augmentations(i, teacher_out, student_out, images, ema_images, batch_size, epoch,labels)

    logger(epoch,epoch_student_dice_per_class,epoch_teacher_dice_per_class,
            model,ema_model,epoch_start,total_epochs,epoch_student_loss,
            epoch_class_loss,epoch_consistency_loss,epoch_teacher_loss, 
            num_batches=len(train_loader), model_save = 10)

    if sweep:
        print("Epoch", epoch, "student_dice_per_class:", epoch_student_dice_per_class)
        if len(epoch_student_dice_per_class) == 6:
            student_avg_dice = sum(epoch_student_dice_per_class[1:]) / 5.0
            teacher_avg_dice = sum(epoch_teacher_dice_per_class[1:]) / 5.0
        else:
            student_avg_dice = 0.0
            teacher_avg_dice = 0.0

        print(f" epoch: {epoch}, student_dice: {student_avg_dice},teacher_dice: {teacher_avg_dice}, student loss: {epoch_student_loss} ,classification loss : {epoch_class_loss} ,consistency loss : {epoch_consistency_loss},'consistency weight : {consistency_weight}")

        wandb.log({
            "epoch": epoch,
            "student_dice": student_avg_dice,
            "teacher_dice": teacher_avg_dice,
            'student_loss': epoch_student_loss ,
            'classification_loss' : epoch_class_loss ,
            'consistency_loss' : epoch_consistency_loss,
            'consistency_weight' : consistency_weight
        })

    return epoch_class_loss, epoch_consistency_loss, consistency_weight



def get_dice_scores(student_out_valid, teacher_out,true_labels_valid,valid_mask_flat,epoch_student_dice_per_class,epoch_teacher_dice_per_class):
    if student_out_valid.shape[0] > 0:
        student_out_probs = torch.softmax(student_out_valid, dim=1)
        student_dice_list = multiclass_dice_scores(student_out_probs, true_labels_valid, num_classes=6)
        for cls_idx, cls_dice in enumerate(student_dice_list):
            epoch_student_dice_per_class[cls_idx] += cls_dice


    teacher_out_flat = teacher_out.permute(0, 2, 3, 1).reshape(-1, teacher_out.shape[1])
    teacher_out_valid = teacher_out_flat[valid_mask_flat]
    if teacher_out_valid.shape[0] > 0:
        teacher_out_probs = torch.softmax(teacher_out_valid, dim=1)
        teacher_dice_list = multiclass_dice_scores(teacher_out_probs, true_labels_valid, num_classes=6)
        for cls_idx, cls_dice in enumerate(teacher_dice_list):
            epoch_teacher_dice_per_class[cls_idx] += cls_dice

    return epoch_student_dice_per_class, epoch_teacher_dice_per_class


def vis_augmentations(i, teacher_out, student_out, images, ema_images, batch_size, epoch,labels):
    if i == 0:  # Only visualize first batch of each epoch
        with torch.no_grad():
            student_preds = student_out.argmax(dim=1).cpu()  # Convert to class labels
            teacher_preds = teacher_out.argmax(dim=1).cpu()

        # Convert images back to displayable format
        weak_input = images.cpu().permute(0,2,3,1).numpy()  # B, H, W, C
        strong_input = ema_images.cpu().permute(0,2,3,1).numpy()  # B, H, W, C
        labels = labels.cpu().squeeze(1).numpy()  # Ground truth labels

        fig, axs = plt.subplots(4, batch_size, figsize=(batch_size * 2, 8))
        
        for idx in range(batch_size):
            axs[0, idx].imshow((weak_input[idx] * 0.5 + 0.5).clip(0,1))
            axs[0, idx].set_title("Weakly Augmented")
            axs[0, idx].axis("off")

            axs[1, idx].imshow((strong_input[idx] * 0.5 + 0.5).clip(0,1))
            axs[1, idx].set_title("Strongly Augmented")
            axs[1, idx].axis("off")

            axs[2, idx].imshow(student_preds[idx], cmap="tab20")
            axs[2, idx].set_title("Student Prediction")
            axs[2, idx].axis("off")

            axs[3, idx].imshow(teacher_preds[idx], cmap="tab20")
            axs[3, idx].set_title("Teacher Prediction")
            axs[3, idx].axis("off")

        plt.tight_layout()
        plt.savefig(f"visualization_epoch_{epoch}.png")
        plt.close()

def logger(epoch,epoch_student_dice_per_class,epoch_teacher_dice_per_class,
           model,ema_model,epoch_start,total_epochs,epoch_student_loss,
           epoch_class_loss,epoch_consistency_loss,epoch_teacher_loss, num_batches,
           model_save = 10):
    
    epoch_student_loss /= num_batches
    epoch_class_loss /= num_batches
    epoch_consistency_loss /= num_batches
    epoch_teacher_loss /= num_batches

    for cls_idx in range(6):
        epoch_student_dice_per_class[cls_idx] /= num_batches
        epoch_teacher_dice_per_class[cls_idx] /= num_batches
    with open("per_class_dice_log.txt", "a") as f:
        f.write(f"Epoch {epoch+1} - Student per-class dice: ")
        f.write(", ".join([f"{d:.4f}" for d in epoch_student_dice_per_class]))
        f.write("\n")
        f.write(f"Epoch {epoch+1} - Teacher per-class dice: ")
        f.write(", ".join([f"{d:.4f}" for d in epoch_teacher_dice_per_class]))
        f.write("\n\n")

    if (epoch + 1) % model_save == 0:
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
        torch.save(ema_model.state_dict(), f"model_ema_epoch_{epoch+1}.pth")
        print(f"Models saved at epoch {epoch+1}")

    epoch_time = time.time() - epoch_start


    # Print final line with epoch-average values
    print(f"Epoch: {epoch + 1}/{total_epochs} | "
          f"Train Loss: {epoch_student_loss:.4f} | "
          f"Class Loss: {epoch_class_loss:.4f} | "
          f"Consis Loss: {epoch_consistency_loss:.4f} | "
          f"EMA Class Loss: {epoch_teacher_loss:.4f} | "
          f"Student Dice: {epoch_student_dice_per_class} | "
          f"Teacher Dice: {epoch_teacher_dice_per_class} | "
          f"Time: {epoch_time:.2f}s")
    



