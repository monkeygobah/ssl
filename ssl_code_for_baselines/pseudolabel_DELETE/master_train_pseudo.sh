python full_main.py --train \
                     --labeled_img_path 'pl_data/unlabeled_images_all' \
                     --labeled_mask_path pl_data/pseudolabeled_masks_all \
                     --model_name 'all_pseudolabels' \
                     --pretrained_model \
                     --base_model_path 'models/celeb.pth'  \
                     --lr .00001



python full_main.py --train \
                     --labeled_img_path 'pl_data/unlabeled_images_08' \
                     --labeled_mask_path pl_data/pseudolabeled_masks_08 \
                     --model_name '08_pseudolabels' \
                     --pretrained_model \
                     --base_model_path 'models/celeb.pth' \
                     --lr .00001


python full_main.py --train \
                     --labeled_img_path 'pl_data/unlabeled_images_05' \
                     --labeled_mask_path pl_data/pseudolabeled_masks_05 \
                     --model_name '05_pseudolabels' \
                     --pretrained_model \
                     --base_model_path 'models/celeb.pth' \
                     --lr .00001