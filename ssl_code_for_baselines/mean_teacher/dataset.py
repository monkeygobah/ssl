import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage import io
from PIL import Image
import random 

from batch_sampler import TransformTwice


import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from batch_sampler import TwoStreamBatchSampler, SingleStreamBaselineSampler


class LabeledUnlabeledDataset(Dataset):
    def __init__(self, labeled_img_dir, labeled_mask_dir=None, unlabeled_img_dir=None, train=True, transform=None):
        """
        Initialize the dataset.

        Args:
            labeled_img_dir (str): Path to labeled images.
            labeled_mask_dir (str, optional): Path to labeled masks. Required if `train=True`.
            unlabeled_img_dir (str, optional): Path to unlabeled images. Required if `train=True`.
            train (bool): Whether this is training data or test data.
            transform (callable, optional): Transformations to apply to images.
        """
        self.labeled_img_dir = labeled_img_dir
        self.labeled_mask_dir = labeled_mask_dir
        self.unlabeled_img_dir = unlabeled_img_dir
        self.train = train
        self.transform = transform

        # Load file names
        self.labeled_imgs = sorted(os.listdir(labeled_img_dir)) if labeled_img_dir else []
        self.labeled_masks = sorted(os.listdir(labeled_mask_dir)) if labeled_mask_dir else []
        self.unlabeled_imgs = sorted(os.listdir(unlabeled_img_dir)) if unlabeled_img_dir else []

        # Combine labeled and unlabeled images for training
        if train:
            self.total_imgs = self.labeled_imgs + self.unlabeled_imgs
        else:
            self.total_imgs = self.labeled_imgs

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset given an index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            - image: Tensor (C x H x W)
            - ema_image: Augmented version of the image (Tensor)
            - mask: Tensor (H x W) for labeled data or -1 for unlabeled data.
            - true_mask: Tensor (H x W) for labeled data or None for unlabeled data.
        """
        if idx < len(self.labeled_imgs):  # Labeled data
            img_path = os.path.join(self.labeled_img_dir, self.labeled_imgs[idx])
            mask_path = os.path.join(self.labeled_mask_dir, self.labeled_masks[idx]) if self.labeled_mask_dir else None

            # Load image and mask
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path) if mask_path else None

            # Apply noise transform for EMA image BEFORE tensor conversion
            ema_image = self.transform['noise'](image) if 'noise' in self.transform else image

            # Apply the standard image transform
            image = self.transform['image'](image) if 'image' in self.transform else image

            # Transform mask
            mask = self.transform['mask'](mask) if mask and 'mask' in self.transform else None

            return image, ema_image, mask, mask  # true_mask == mask for labeled data
        else:  # Unlabeled data
            unlabeled_idx = idx - len(self.labeled_imgs)
            img_path = os.path.join(self.unlabeled_img_dir, self.unlabeled_imgs[unlabeled_idx])

            # Load image
            image = Image.open(img_path).convert("RGB")

            # Apply noise transform for EMA image BEFORE tensor conversion
            ema_image = self.transform['noise'](image) if 'noise' in self.transform else image

            # Apply the standard image transform
            image = self.transform['image'](image) if 'image' in self.transform else image

            # Return empty masks for unlabeled data
            empty_mask = torch.full((1, 256, 256), -1, dtype=torch.float32)  # Ensure it's a Tensor
            return image, ema_image, empty_mask, empty_mask


def get_transforms():
    resize_size = (256, 256)

    image_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    noise_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0))
    ])

    mask_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=Image.NEAREST),
        transforms.PILToTensor()
    ])

    return {"image": image_transform, "noise": noise_transform, "mask": mask_transform}



def dataloader_setup(batch_size,labeled_batch_size):
    labeled_img_dir = '../../data/preprocessed_train/labeled_images_processed'
    labeled_mask_dir = '../../data/preprocessed_train/labeled_masks_processed'
    unlabeled_img_dir = '../../data/unlabeled_images_processed_small'
    transforms_dict = get_transforms()

    train_data = LabeledUnlabeledDataset(
        labeled_img_dir=labeled_img_dir,
        labeled_mask_dir=labeled_mask_dir,
        unlabeled_img_dir=unlabeled_img_dir,
        train=True,
        transform=transforms_dict
    )


    labeled_idxs = list(range(len(train_data.labeled_imgs)))
    unlabeled_idxs = list(range(len(train_data.labeled_imgs), len(train_data.total_imgs)))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, labeled_batch_size)
    train_loader = torch.utils.data.DataLoader(train_data, batch_sampler=batch_sampler, num_workers=0, pin_memory=True)
    return train_loader


