import os
import numpy as np
from PIL import Image

# Paths
pseudo_mask_dir = "pseudomasks"  # Adjust if needed
unlabeled_img_dir = "/root/ssl/data/unlabeled_images_processed"

# Check mask files
mask_files = sorted(os.listdir(pseudo_mask_dir))
img_files = sorted(os.listdir(unlabeled_img_dir))

num_classes = 6  # Expected number of segmentation classes
errors = []

for mask_file in mask_files:
    mask_path = os.path.join(pseudo_mask_dir, mask_file)
    mask = Image.open(mask_path)
    
    # Convert to numpy
    mask_np = np.array(mask)

    # Check shape consistency
    if mask_np.shape != (256, 256):  # Modify if your images have a different resolution
        errors.append(f"❌ Shape mismatch: {mask_file} has shape {mask_np.shape}")

    # Check valid class values
    unique_vals = np.unique(mask_np)
    if not np.all(np.isin(unique_vals, np.arange(num_classes))):
        errors.append(f"❌ Invalid label values in {mask_file}: Found {unique_vals}")

    # Check if corresponding image exists
    img_filename = os.path.splitext(mask_file)[0] + ".jpg"  # Adjust extension if needed
    img_path = os.path.join(unlabeled_img_dir, img_filename)
    
    if not os.path.exists(img_path):
        errors.append(f"❌ Missing image for mask: {mask_file}")

# Print results
if errors:
    print("\n".join(errors))
    print(f"\n❌ {len(errors)} issues found!")
else:
    print("✅ All pseudo-masks are valid for training!")
