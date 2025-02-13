import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet101
import wandb

# def add_noise(images):
#     """
#     Add noise to images
#     Args:
#         images (torch.Tensor): images
#     Returns:
#         noisy_images (torch.Tensor): noisy images
#     """

#     # Adding non destructive noise to images - structure of image remains the same but the pixel values are changed
#     noisy_images = transforms.ColorJitter(
#         brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(images)
#     # noisy_images = transforms.RandomGrayscale(p=0.2)(noisy_images)
#     noisy_images = transforms.GaussianBlur(3, sigma=(0.1, 2.0))(noisy_images)
#     return noisy_images

def softmax_mse_loss(input, target):
    """
    Compute the softmax cross entropy loss
    Args:
        input (torch.Tensor): input
        target (torch.Tensor): target
    Returns:
        torch.Tensor: softmax cross entropy loss
    """
    return torch.mean(torch.sum(F.softmax(input, dim=1) * F.mse_loss(input, target, reduction='none'), dim=1))

def update_ema(model, ema_model, alpha, global_step):
    """
    Update the ema model weights with the model weights
    Args:
        model (torch.nn.Module): model
        ema_model (torch.nn.Module): ema model
        alpha (float): alpha
        global_step (int): global step
    """
    
    # Set alpha to 0.999 at the beginning and then linearly decay
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


# def dice_binary(ps, ts):
#     ps = torch.round(ps).to(ps.dtype)
#     ts = torch.round(ts).to(ts.dtype)
#     return dice_score(ps, ts)


# def dice_score(ps, ts, eps=1e-7):
#     numerator = torch.sum(ts * ps, dim=(1, 2, 3)) * 2 + eps
#     denominator = torch.sum(ts, dim=(1, 2, 3)) + \
#         torch.sum(ps, dim=(1, 2, 3)) + eps
#     return numerator / denominator


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)

    # Cast target to long
    target = target.long()

    loss = F.cross_entropy(
        input, target, weight=weight, reduction='mean', ignore_index=-1
    )
    return loss


import torch
def multiclass_dice_scores(pred_probs, target_labels, num_classes=6, eps=1e-6):
    # Flatten if 4D
    if pred_probs.dim() == 4:
        B, C, H, W = pred_probs.shape
        pred_probs = pred_probs.permute(0, 2, 3, 1).reshape(-1, C)
        target_labels = target_labels.reshape(-1)
    elif pred_probs.dim() == 2:
        pass
    else:
        raise ValueError(f"pred_probs must be 2D or 4D, got shape {pred_probs.shape}.")

    # Hard predictions via argmax
    pred_classes = torch.argmax(pred_probs, dim=1)  # [N]
    
    dice_list = []
    for cls in range(num_classes):
        pred_mask = (pred_classes == cls).float()
        target_mask = (target_labels == cls).float()

        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        if union > 0:
            dice_val = (2.0 * intersection + eps) / (union + eps)
            dice_list.append(dice_val.item())
        else:
            # If class cls not present in union, skip or treat as 1.0 or 0.0
            # We'll treat it as 1.0 (perfect) if neither pred nor GT contain that class.
            dice_list.append(1.0 if intersection == 0 and union == 0 else 0.0)
    return dice_list


def visualize_predictions(images, preds, labels, max_samples=2):
    for i in range(min(max_samples, images.shape[0])):
        img = images[i].cpu().permute(1,2,0).numpy()  # [H, W, 3]
        pred_mask = preds[i].cpu().numpy()            # [H, W]
        label_mask = labels[i].cpu().numpy().squeeze(0)  # [H, W] (removes extra dim)

        fig, axs = plt.subplots(1,3, figsize=(12,4))
        axs[0].imshow((img*0.5 + 0.5).clip(0,1))
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        axs[1].imshow(pred_mask, cmap="tab20")
        axs[1].set_title("Predicted Classes")
        axs[1].axis("off")

        axs[2].imshow(label_mask, cmap="tab20")
        axs[2].set_title("Ground Truth")
        axs[2].axis("off")

        plt.savefig(f"batch_{i}_visual.png")


def plot_train_metrics(epochs,supervised_losses,consistency_losses,consistency_weights):
    plt.figure(figsize=(8,6))
    plt.plot(epochs, supervised_losses, label="Supervised Loss", marker='o')
    plt.plot(epochs, consistency_losses, label="Consistency Loss", marker='o')
    plt.plot(epochs, consistency_weights, label="Consistency Weight", linestyle='dashed', color='gray')

    plt.xlabel("Epoch")
    plt.ylabel("Loss / Weight")
    plt.title("Consistency vs Supervised Loss Over Time")
    plt.legend()
    plt.grid()
    plt.savefig('training_loss_metrics.png')


def build_models():
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {DEVICE}')

    # Initialize the model
    model = deeplabv3_resnet101(num_classes=6)
    model.to(DEVICE)

    model_ema = deeplabv3_resnet101(num_classes=6)
    model_ema.to(DEVICE)

    # Detach the EMA model parameters from the graph to prevent backprop
    for param in model_ema.parameters():
        param.detach_() 
    return model, model_ema


def wandb_setup(batch_size,labeled_batch_size,learning_rate,alpha,consistency, cons_style, consistency_rampup):
    wandb.init(project="mean-teacher-segmentation")

    # Define hyperparameters
    wandb.config.update({
        "batch_size": batch_size,
        "labeled_batch_size": labeled_batch_size,
        "learning_rate": learning_rate,
        "alpha": alpha,
        "consistency_weight": consistency,
        "rampup_style": cons_style,
        "consistency rampup" : consistency_rampup
    })
