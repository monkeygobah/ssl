          

import os
import time
import torch
import datetime
import wandb 
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional as F
from full_data_loader import Data_Loader_Split
from full_utils import *
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.detection import maskrcnn_resnet50_fpn



def dice_coefficient(pred, target, class_index=1, epsilon=1e-6):
    pred_cls = (pred == class_index).float()
    target_cls = (target == class_index).float()
    
    intersection = (pred_cls * target_cls).sum(dim=[1, 2])
    union = pred_cls.sum(dim=[1, 2]) + target_cls.sum(dim=[1, 2])
    
    dice_score = (2. * intersection + epsilon) / (union + epsilon)
    return dice_score.mean()



def visualize_batch(images, predicted_labels, true_labels):
    batch_size = images.shape[0]
    fig, axs = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
    for i in range(batch_size):
        axs[i, 0].imshow(images[i].permute(1, 2, 0).cpu().numpy())
        axs[i, 0].set_title('Input Image')
        axs[i, 1].imshow(predicted_labels[i].cpu().numpy(), cmap='gray')
        axs[i, 1].set_title('Predicted Labels')
        axs[i, 2].imshow(true_labels[i].cpu().numpy(), cmap='gray')
        axs[i, 2].set_title('True Labels')
        for ax in axs[i]:
            ax.axis('off')
    plt.savefig('batch_visualized_dlv3.jpg')

class Trainer(object):
    def __init__(self, data_loader, config, hp_tune = None,device='cpu'):

        # Data loader
        self.data_loader = data_loader

        self.imsize = config.imsize
        self.parallel = config.parallel

        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.lr = config.lr

        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.img_path = config.labeled_img_path
        self.label_path = config.labeled_mask_path 

        self.train_switch = config.train
    
        self.model_save_path = config.model_save_path
        
        # self.sample_path = config.sample_path
        
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.model_name =config.model_name
        
        self.model_save_path = os.path.join(config.model_save_path)

        self.w_b_config = hp_tune        
        self.name = config.name
        
        self.device = device

        self.base_model_path = config.base_model_path
        self.build_model()

        # Start with trained model
        if self.pretrained_model:
            self.G.load_state_dict(torch.load(self.base_model_path))

    def train(self):
        print(f"Using device: {self.device}")
        # Data iterator
        data_iter = iter(self.data_loader)
        # step_per_epoch = len(self.data_loader)
        
        model_save_step = int(self.model_save_step)
        
        start = 0

        # Start time
        start_time = time.time()
        
        for step in range(start, self.total_step):
            self.G.train()
            
            try:
                imgs, labels = next(data_iter)
                # print(imgs[0].shape)
                    
            except:
                data_iter = iter(self.data_loader)
                imgs, labels = next(data_iter)

            size = labels.size()

            imgs = imgs.to(self.device)

            # ================== Train G =================== #
            # Process labels for both UNet and DeepLabV3 in the same way
            labels[:, 0, :, :] = labels[:, 0, :, :] * 255.0  
            labels_real_plain = labels[:, 0, :, :].to(self.device)  

            # Convert labels for one-hot encoding even though it's only needed for visualization/comparison
            labels = labels[:, 0, :, :].view(size[0], 1, size[2], size[3])
            oneHot_size = (size[0], 6, size[2], size[3])
            
            labels_real = torch.FloatTensor(torch.Size(oneHot_size)).zero_().to(self.device)
            labels_real = labels_real.scatter_(1, labels.data.long().to(self.device), 1.0)
            
        
            labels_predict = self.G(imgs)['out'] 
     

            # print("Labels predict shape:", labels_predict.shape)
            # print("Labels predict unique values:", torch.unique(labels_predict))   
            # break
            # Compute loss
  
            c_loss = cross_entropy2d(labels_predict, labels_real_plain.long())
            
            # labels_predict_dice = labels_predict.argmax(dim=1)


            # # Calculate cross entropy loss
            # print(f'CEL: {c_loss}')

            ####### MAKE DICE SCORE COEFFICIENT MAKE SENSE FOR BATCHES HERE
            # dice_score_class_1 = dice_coefficient(labels_predict_dice, labels_real_plain.long(), class_index=1)
            # dice_score_class_2 = dice_coefficient(labels_predict_dice, labels_real_plain.long(), class_index=2)
            # dice_score_class_3 = dice_coefficient(labels_predict_dice, labels_real_plain.long(), class_index=3)
            # dice_score_class_4 = dice_coefficient(labels_predict_dice, labels_real_plain.long(), class_index=4)
            # dice_score_class_5 = dice_coefficient(labels_predict_dice, labels_real_plain.long(), class_index=5)
            


            # print(f"DICE SCORE FOR CLASS 1 : {dice_score_class_1.item()}")
            # print(f"DICE SCORE FOR CLASS 2 : {dice_score_class_2.item()}")
            # print(f"DICE SCORE FOR CLASS 3 : {dice_score_class_3.item()}")
            # print(f"DICE SCORE FOR CLASS 4 : {dice_score_class_4.item()}")
            # print(f"DICE SCORE FOR CLASS 5 : {dice_score_class_5.item()}")


            # num_plots = min(5, self.batch_size)
            # fig, axs = plt.subplots(num_plots, 2, figsize=(10, 5*num_plots))

            # for i in range(num_plots):
            #     # Plot predicted labels
            #     axs[i, 0].imshow(labels_predict_dice[i].cpu().numpy(), cmap='viridis')
            #     axs[i, 0].set_title(f'Predicted Labels (Batch {i+1})')
            #     axs[i, 0].axis('off')
                
            #     # Plot ground truth labels
            #     axs[i, 1].imshow(labels_real_plain[i].cpu().numpy(), cmap='viridis')
            #     axs[i, 1].set_title(f'Ground Truth Labels (Batch {i+1})')
            #     axs[i, 1].axis('off')

            # plt.tight_layout()            
            # plt.savefig(f'{step}_iris_log.jpg')
        
            self.reset_grad()
            c_loss.backward()
            self.g_optimizer.step()
            
            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], Cross_entrophy_loss: {:.4f}".
                      format(elapsed, step + 1, self.total_step, c_loss.data))

            label_batch_predict = generate_label(labels_predict, self.imsize)
            label_batch_real = generate_label(labels_real, self.imsize)

            img_combine = imgs[0]
            real_combine = label_batch_real[0]
            predict_combine = label_batch_predict[0]
            for i in range(1, len(imgs)):
                img_combine = torch.cat([img_combine, imgs[i]], 2)
                real_combine = torch.cat([real_combine, label_batch_real[i]], 2)
                predict_combine = torch.cat([predict_combine, label_batch_predict[i]], 2)


            if (step + 1) % self.sample_step == 0:
                labels_sample = self.G(imgs)['out']
        
                labels_sample = generate_label(labels_sample, self.imsize)

                
            if (step+1) % model_save_step==0:
                torch.save(self.G.state_dict(),
                            os.path.join(self.model_save_path, f'{step + 1}_{self.model_name}.pth'))



    def build_model(self):
        self.G = deeplabv3_resnet101(pretrained=True)
        self.G.classifier = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 6, kernel_size=(1, 1), stride=(1, 1))
        )
        self.G.to(self.device)
            
        if self.parallel:
            self.G = nn.DataParallel(self.G)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.lr, [self.beta1, self.beta2])

    def reset_grad(self):
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
        
        
        

