
import os
import time
import torch
import datetime
import numpy as np
import pickle
from scipy.ndimage import label

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import PIL

from full_test_utils import *

import re
from full_utils import *
from PIL import Image, ImageOps

import pandas as pd
import csv
from torchvision.models.segmentation import deeplabv3_resnet101


def apply_color_map(mask, color_map):
    """
    Applies a color map to a binary mask, assigning specific colors to different class labels.

    Parameters:
        mask (np.array): A binary mask where each pixel is labeled with a class.
        color_map (dict): A dictionary mapping class labels to RGB color values.

    Returns:
        np.array: A colored mask with shape (height, width, 3).
    """
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for value, color in color_map.items():
        colored_mask[mask == value] = color
    
    return colored_mask

def overlay_mask_on_image(image, mask, alpha=0.5):
    """
    Overlays a colored mask onto an image with a specified transparency.

    Parameters:
        image (PIL.Image): The original image.
        mask (np.array): The mask to overlay, where each pixel is labeled with a class.
        alpha (float, optional): The transparency level of the overlay (0 = fully transparent, 1 = fully opaque). Default is 0.5.

    Returns:
        PIL.Image: The image with the colored mask overlay.
    """
    color_map = {
        0: [255, 0, 0],    # Red for background (or class 0)
        1: [0, 255, 0],    # Green for class 1
        2: [0, 0, 255],    # Blue for class 2
        3: [255, 255, 0],  # Yellow for class 3
        4: [255, 0, 255],  # Magenta for class 4
        5: [0, 255, 255],  # Cyan for class 5
    }
    colored_mask = apply_color_map(mask, color_map)
    image_np = np.array(image).astype(float)
    overlay = np.where(mask[..., None] > 0, image_np * (1 - alpha) + colored_mask * alpha, image_np)

    return Image.fromarray(overlay.astype(np.uint8))

def transformer(dynamic_resize_and_pad, totensor, normalize, centercrop, imsize, is_mask=False):
    options = []
    if centercrop:
        options.append(transforms.CenterCrop(160))
    if dynamic_resize_and_pad:
        options.append(ResizeAndPad(output_size=(imsize, imsize)))
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(options)


class ResizeAndPad:
    def __init__(self, output_size=(512, 512), fill=0, padding_mode='constant'):
        self.output_size = output_size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        # Calculate new height maintaining aspect ratio
        original_width, original_height = img.size
        new_height = int(original_height * (self.output_size[0] / original_width))
        img = img.resize((self.output_size[0], new_height), Image.NEAREST)

        # Calculate padding
        padding_top = (self.output_size[1] - new_height) // 2
        padding_bottom = self.output_size[1] - new_height - padding_top

        # Apply padding
        img = ImageOps.expand(img, (0, padding_top, 0, padding_bottom), fill=self.fill)
        
        return img
    

class Tester(object):
    def __init__(self, config, device):
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

        self.unlabeled_img_path = config.unlabeled_img_path
        # self.unlabeled_label_path = config.unlabeled_mask_path   

        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # self.version = config.version
        self.device = device
        self.model_save_path = config.model_save_path

        # self.dlv3 = config.dlv3
        self.test_image_path = config.test_image_path
        self.test_label_path = config.test_label_path

        self.test_size = config.test_size
        self.model_name = config.model_name
        self.csv_path = config.csv_path

        self.name= config.name

        self.build_model()

    def test(self):
        transform = transform_img_split(resize=True, totensor=True, normalize=True)
        
        transform_gt = transformer(dynamic_resize_and_pad=True, totensor=True, normalize=False, centercrop=False, imsize=512)
        transform_gt_plot = transformer(dynamic_resize_and_pad=True, totensor=False, normalize=False, centercrop=False, imsize=512)

        test_paths_imgs = make_dataset(self.test_image_path)
        test_paths_labels = make_dataset(self.test_label_path, gt=True)
        
        test_paths_imgs_forward, test_paths_labels_forward = align_image_and_label_paths(test_paths_imgs, test_paths_labels)

        self.G.load_state_dict(torch.load(os.path.join('models', self.model_name)))
        self.G.eval() 

        batch_num = int(self.test_size / self.batch_size)
        storage = []
        names = []

        for i in range(batch_num):
            imgs = []
            gt_labels = []
            l_imgs = []
            r_imgs = []
            original_sizes = []
            
            for j in range(self.batch_size):
                current_idx = i * self.batch_size + j
                if current_idx < len(test_paths_imgs_forward):
                    path = test_paths_imgs_forward[current_idx]
                    name = path.split('/')[-1][:-4]
                    names.append(name)

                    original_sizes.append(Image.open(path).size)
                    l_img, r_img = transform(Image.open(path))
                    l_imgs.append(l_img)
                    r_imgs.append(r_img)
                    
                    gt_path = test_paths_labels_forward[current_idx]
                    gt_img = Image.open(gt_path)
                    gt_labels.append(transform_gt(gt_img).numpy())

                else:
                    break 
                
            if len(l_imgs) != 0:
                labels_predict_plain = self.predict_split_face(l_imgs, r_imgs, self.imsize, transform_gt_plot, self.device, original_sizes, self.G)                        
                visualize_and_dice(labels_predict_plain, np.array(gt_labels), names, storage)
                names = [] 


        write_dice_scores_to_csv(storage, self.csv_path, self.name)



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
 

    def predict_split_face(self, l_imgs,r_imgs, imsize, transform_plotting, device, original_sizes, G):
        l_imgs = torch.stack(l_imgs) 
        r_imgs = torch.stack(r_imgs) 
        
        
        l_imgs = l_imgs.to(device)
        r_imgs = r_imgs.to(device)
        
        l_labels_predict = G(l_imgs)['out'] 
        r_labels_predict = G(r_imgs)['out']  
        
        
        l_labels_predict_plain = generate_label_plain(l_labels_predict, imsize)
        r_labels_predict_plain = generate_label_plain(r_labels_predict, imsize)
        
        labels_predict_plain = []

        for idx, (left_pred, right_pred) in enumerate(zip(l_labels_predict_plain, r_labels_predict_plain)):
            original_width, original_height = original_sizes[idx]
            mid = original_width // 2
            
            left_width = mid  
            right_width = original_width - mid  

            left_pred_resized = cv2.resize(left_pred, (left_width, original_height), interpolation=cv2.INTER_NEAREST)
            right_pred_resized = cv2.resize(right_pred, (right_width, original_height), interpolation=cv2.INTER_NEAREST)

            stitched = np.zeros((original_height, original_width), dtype=np.uint8)

            stitched[:, :mid] = left_pred_resized
            stitched[:, mid:] = right_pred_resized
        
            
            resized_stitched = transform_plotting(Image.fromarray(stitched))
            labels_predict_plain.append(np.array(resized_stitched))
            
            # plt.imshow(np.array(resized_stitched))
            # plt.savefig('test.jpg')
            
            
            # # Load the original image for overlay
            # original_image = Image.open(os.path.join(self.test_image_path, f'{names[idx]}.jpg'))

            # # Overlay the mask on the original image
            # overlaid_image = overlay_mask_on_image(original_image, np.array(resized_stitched), color_map)

            # # Save the overlaid image
            # save_path = os.path.join(save_dir, f'overlaid_{idx}.png')
            # overlaid_image.save(save_path)
            

            
            
        labels_predict_plain = np.array(labels_predict_plain)    
        return labels_predict_plain

    def generate_pseudolabels(self, pseudolabel_save_path, confidence_threshold=None):
        """
        Generate pseudolabels for all unlabeled images and save them, using split, resize, and stitching logic.

        Args:
            pseudolabel_save_path (str): Path to save pseudolabeled masks.
            confidence_threshold (float, optional): Threshold for pixel-wise confidence. Defaults to None.
        """
        self.G.eval()
        self.G.load_state_dict(torch.load(os.path.join('models', self.model_name)))
 
        os.makedirs(pseudolabel_save_path, exist_ok=True)
        
        # Load the unlabeled image paths
        unlabeled_paths = make_dataset(self.unlabeled_img_path)

        # Define transformations
        transform_split = transform_img_split(resize=True, totensor=True, normalize=True)
        transform_plotting = transformer(dynamic_resize_and_pad=True, totensor=False, normalize=False, centercrop=False, imsize=self.imsize)

        for i, img_path in enumerate(unlabeled_paths):
            img_name = os.path.basename(img_path)
            img = Image.open(img_path)

            # Split and transform the left and right halves of the image
            l_img, r_img = transform_split(img)

            l_imgs = torch.stack([l_img]).to(self.device)
            r_imgs = torch.stack([r_img]).to(self.device)

            with torch.no_grad():
                l_output = self.G(l_imgs)['out']
                r_output = self.G(r_imgs)['out']

                l_probs = torch.softmax(l_output, dim=1)
                r_probs = torch.softmax(r_output, dim=1)

                l_confidence, l_pseudolabel = torch.max(l_probs, dim=1)
                r_confidence, r_pseudolabel = torch.max(r_probs, dim=1)

            if confidence_threshold is not None:
                l_high_confidence = (l_confidence.squeeze(0) >= confidence_threshold)
                r_high_confidence = (r_confidence.squeeze(0) >= confidence_threshold)

                # Match dimensions
                l_pseudolabel = l_pseudolabel.squeeze(0)
                r_pseudolabel = r_pseudolabel.squeeze(0)

                l_pseudolabel[~l_high_confidence] = 0
                r_pseudolabel[~r_high_confidence] = 0


            original_width, original_height = img.size
            mid = original_width // 2

            left_width = mid
            right_width = original_width - mid

            l_pseudolabel_resized = cv2.resize(
                l_pseudolabel.squeeze(0).cpu().numpy(),
                (left_width, original_height),
                interpolation=cv2.INTER_NEAREST
            )
            r_pseudolabel_resized = cv2.resize(
                r_pseudolabel.squeeze(0).cpu().numpy(),
                (right_width, original_height),
                interpolation=cv2.INTER_NEAREST
            )

            stitched_pseudolabel = np.zeros((original_height, original_width), dtype=np.uint8)
            stitched_pseudolabel[:, :mid] = l_pseudolabel_resized
            stitched_pseudolabel[:, mid:] = r_pseudolabel_resized

            # Save the stitched pseudolabel
            save_path = os.path.join(pseudolabel_save_path, img_name[:-3] + '.png')
            Image.fromarray(stitched_pseudolabel).save(save_path)

            # Visualize the first few pseudolabels
            if i < 5:
                overlayed = overlay_mask_on_image(img, stitched_pseudolabel)
                os.makedirs("visualizations", exist_ok=True)
                overlayed.save(f"visualizations/{os.path.basename(img_path)}")

            # Optional progress log
            if i % 100 == 0:
                print(f"Processed {i}/{len(unlabeled_paths)} images...")
