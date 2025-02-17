import os
import shutil

# Paths to folders
pseudolabel_folders = [
    "pseudolabeled_masks_all",
    "pseudolabeled_masks_08",
    "pseudolabeled_masks_05"
]
unlabeled_images_folder = "../data/unlabeled_images"
output_base_folder = "matched_images"

if not os.path.exists(output_base_folder):
    os.makedirs(output_base_folder)

unlabeled_images = {os.path.splitext(f)[0]: f for f in os.listdir(unlabeled_images_folder) if os.path.isfile(os.path.join(unlabeled_images_folder, f))}

# Process each pseudolabel folder
for pseudolabel_folder in pseudolabel_folders:
    print(f"Processing folder: {pseudolabel_folder}")
    output_folder = os.path.join(output_base_folder, os.path.basename(pseudolabel_folder))
    os.makedirs(output_folder, exist_ok=True)

    for mask_file in os.listdir(pseudolabel_folder):
        if not mask_file.endswith('.png'):
            continue

        corrected_name = mask_file.replace("..png", ".png")
        mask_path = os.path.join(pseudolabel_folder, mask_file)
        corrected_path = os.path.join(pseudolabel_folder, corrected_name)

        if mask_file != corrected_name:
            os.rename(mask_path, corrected_path)

        base_name = os.path.splitext(corrected_name)[0]
        if base_name in unlabeled_images:
            image_file = unlabeled_images[base_name]
            src_image_path = os.path.join(unlabeled_images_folder, image_file)
            dest_image_path = os.path.join(output_folder, image_file)

            # Copy the matched image to the output folder
            shutil.copy(src_image_path, dest_image_path)

print("Processing complete!")
