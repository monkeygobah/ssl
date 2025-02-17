import torch
import torchvision.datasets as dsets
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

NUM_WORKERS = 0

def visualize_dataset(images, labels, idx,num_images=5):
    fig, axs = plt.subplots(num_images, 2, figsize=(10, num_images * 5))

    for i in range(min(num_images, len(images))):  

        img = images[i].detach().cpu().numpy().transpose((1, 2, 0))
        print(labels[i].size)
        label = labels[i].detach().cpu().numpy().transpose((1, 2, 0))
        axs[i, 0].imshow((img * 0.5 + 0.5))  

        axs[i, 0].set_title('Image')
        axs[i, 1].imshow((label * 0.5 + 0.5).squeeze(), cmap='gray') 
        axs[i, 1].set_title('Label')
        for ax in axs[i]:
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'test_{idx}.jpg')


class DatasetMaker():
    def __init__(self, img_path, label_path, transform_img, transform_label, mode):
        self.img_path = img_path
        self.label_path = label_path
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        
        self.preprocess()

        if mode:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        image_names = sorted([name for name in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path, name))])
        label_names = sorted([name for name in os.listdir(self.label_path) if os.path.isfile(os.path.join(self.label_path, name))])

        for img_name, lbl_name in zip(image_names, label_names):
            img_base_name = os.path.splitext(img_name)[0]  
            lbl_base_name = os.path.splitext(lbl_name)[0]

            if img_base_name == lbl_base_name:
                if self.mode:
                    self.train_dataset.append((os.path.join(self.img_path,img_name),os.path.join(self.label_path, lbl_name)))

            else:
                print(f"Warning: Image {img_base_name} and label {lbl_base_name} do not match!")

        
    def __getitem__(self, index):
        original_index = index // 2
        side = index % 2  

        dataset = self.train_dataset if self.mode else self.test_dataset
        img_path, label_path = dataset[original_index]

        image = Image.open(img_path)
        label = Image.open(label_path)
        
        left_image, right_image = self.transform_img(image)
        left_label, right_label = self.transform_label(label)

        if side == 0:  
            return left_image, left_label
        
        else:  
            return right_image, right_label

    def __len__(self):
        return self.num_images * 2


class Data_Loader_Split():
    def __init__(self, img_path, label_path, image_size, batch_size, mode):
        self.img_path = img_path
        self.label_path = label_path
        self.imsize = image_size
        self.batch = batch_size
        self.mode = mode

    @staticmethod
    def crop_and_resize(img):
        mid = img.width // 2
        left_half = img.crop((0, 0, mid, img.height))
        right_half_start = mid if img.width % 2 == 0 else mid + 1
        right_half = img.crop((right_half_start, 0, img.width, img.height))
        left_resized = left_half.resize((256, 256), Image.NEAREST)
        right_resized = right_half.resize((256, 256), Image.NEAREST)
        return left_resized, right_resized
    
    def transform_img_split(self, resize, totensor, normalize):
        options = []

        if resize:
            options.append(transforms.Lambda(self.crop_and_resize))

        if totensor:
            options.append(transforms.Lambda(lambda imgs: (transforms.ToTensor()(imgs[0]), transforms.ToTensor()(imgs[1]))))
            
        if normalize:
            options.append(transforms.Lambda(lambda imgs: (transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(imgs[0]), 
                                                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(imgs[1]))))
            
        transform = transforms.Compose(options)
        return transform

    def transform_label_split(self, resize, totensor, normalize):
        options = []

        if resize:
            options.append(transforms.Lambda(self.crop_and_resize))

        if totensor:
            options.append(transforms.Lambda(lambda labels: (transforms.ToTensor()(labels[0]), transforms.ToTensor()(labels[1]))))
            
        if normalize:
            options.append(transforms.Lambda(lambda labels: (transforms.Normalize((0, 0, 0), (0, 0, 0))(labels[0]), 
                                                                                transforms.Normalize((0, 0, 0), (0, 0, 0))(labels[1]))))
                
        transform = transforms.Compose(options)
        return transform


    def loader(self):
        transform_img = self.transform_img_split(True, True, True) 
        transform_label = self.transform_label_split(True, True, False)  
        dataset = DatasetMaker(self.img_path, self.label_path, transform_img, transform_label, self.mode)

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch,
                                             shuffle=True,
                                             num_workers=NUM_WORKERS,
                                             drop_last=False)
        
        
        # for idx,(imgs, labels) in enumerate(loader):
        #     print(imgs.size(2), imgs.size(3))
        #     print(labels.size(2), labels.size(3))

        #     # # Visualize the first few images and labels
        #     visualize_dataset(imgs, labels, idx, num_images=2)
        #     if idx==2:
        #         break 
        
        return loader

