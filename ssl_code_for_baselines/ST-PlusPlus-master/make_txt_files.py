import os

# Define dataset directories
paths = {
    "labeled": {
        "images": "/home/georgie/ssl/data/preprocessed_train/labeled_images_processed",
        "masks": "/home/georgie/ssl/data/preprocessed_train/labeled_masks_processed",
        "output": "labeled_ids.txt"
    },
    "unlabeled": {
        "images": "/home/georgie/ssl/data/unlabeled_images_processed",
        "output": "unlabeled_ids.txt"
    },
    "test": {
        "images": "/home/georgie/ssl/data/full_test_cfd/labeled_images_test",
        "masks": "/home/georgie/ssl/data/full_test_cfd/labeled_masks_test",
        "output": "labeled_ids_test.txt"
    }
}

def generate_list(images_dir, masks_dir=None, output_file=None):
    """Generate a list of full paths for images (and masks, if available)."""
    if output_file is None:
        return
    
    image_files = sorted(os.listdir(images_dir))
    
    with open(output_file, "w") as f:
        for img in image_files:
            img_path = os.path.join(images_dir, img)
            
            if masks_dir:  # If masks exist, match by filename
                mask_path = os.path.join(masks_dir, img.replace(".jpg", ".png"))
                if os.path.exists(mask_path):
                    f.write(f"{img_path} {mask_path}\n")
                else:
                    print(f"Warning: No matching mask for {img_path}")
            else:
                f.write(f"{img_path}\n")

# Generate files
generate_list(paths["labeled"]["images"], paths["labeled"]["masks"], paths["labeled"]["output"])
generate_list(paths["unlabeled"]["images"], None, paths["unlabeled"]["output"])
generate_list(paths["test"]["images"], paths["test"]["masks"], paths["test"]["output"])

print("TXT files generated successfully.")
