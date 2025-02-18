import os

# Paths
pseudo_mask_dir = "pseudomasks_best_model"
reliable_ids_path = "reliable_ids/reliable_ids.txt"

# Read reliable image paths
with open(reliable_ids_path, "r") as f:
    reliable_image_paths = [line.strip() for line in f.readlines()]

updated_lines = []

for img_path in reliable_image_paths:
    img_filename = os.path.basename(img_path)
    pseudo_mask_filename = os.path.splitext(img_filename)[0] + ".png"
    pseudo_mask_path = os.path.join(pseudo_mask_dir, pseudo_mask_filename)

    if os.path.exists(pseudo_mask_path):
        updated_lines.append(f"{img_path} {pseudo_mask_path}")
    else:
        print(f"Warning: No pseudo-mask found for {img_path}")

# Write updated reliable_ids.txt
with open(reliable_ids_path, "w") as f:
    f.write("\n".join(updated_lines) + "\n")

print("✅ Updated reliable_ids.txt to include pseudo-mask paths!")
