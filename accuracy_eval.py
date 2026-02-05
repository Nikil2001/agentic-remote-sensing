import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

# CONFIG
pred_dir = "predicted_masks"
gt_dir = "evaluation_masks"
num_classes = 7
mask_size = (128, 128)

# RGB to class index mapping (from your class_dist file)
color_to_class = {
    (0, 255, 255): 0,   # urban_land
    (255, 255, 0): 1,   # agriculture_land
    (255, 0, 255): 2,   # rangeland
    (0, 255, 0): 3,     # forest_land
    (0, 0, 255): 4,     # water
    (255, 255, 255): 5, # barren_land
    (0, 0, 0): 6        # unknown
}

def rgb_to_class(mask_img):
    mask = np.array(mask_img)
    class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for rgb, cls in color_to_class.items():
        match = np.all(mask == rgb, axis=-1)
        class_mask[match] = cls
    return class_mask

def compute_metrics(pred_mask, true_mask, num_classes):
    iou_list = []
    for cls in range(num_classes):
        pred_inds = (pred_mask == cls)
        true_inds = (true_mask == cls)
        intersection = np.logical_and(pred_inds, true_inds).sum()
        union = np.logical_or(pred_inds, true_inds).sum()
        iou = (intersection / union) if union != 0 else 0
        iou_list.append(iou)
    mean_iou = np.mean(iou_list)
    pixel_acc = (pred_mask == true_mask).sum() / true_mask.size
    return mean_iou, pixel_acc

# Evaluation
results = []
print("📁 Current directory:", os.getcwd())
print("🔍 Scanning folders...")

for pred_file in tqdm(sorted(os.listdir(pred_dir))):
    if not pred_file.endswith("_pred.png"):
        continue

    base = pred_file.replace("_pred.png", "")
    gt_file = f"{base}_mask.png"
    pred_path = os.path.join(pred_dir, pred_file)
    gt_path = os.path.join(gt_dir, gt_file)

    if not os.path.exists(gt_path):
        print(f"⚠️ Missing ground truth for: {gt_file}")
        continue

    pred_mask = np.array(Image.open(pred_path).convert("L").resize(mask_size, Image.NEAREST))
    gt_rgb = Image.open(gt_path).resize(mask_size, Image.NEAREST)
    true_mask = rgb_to_class(gt_rgb)

    mean_iou, pixel_acc = compute_metrics(pred_mask, true_mask, num_classes)
    results.append((base, round(mean_iou, 4), round(pixel_acc * 100, 2)))

# Save results
df = pd.DataFrame(results, columns=["Image", "Mean IoU", "Pixel Accuracy (%)"])
df.to_csv("corrected_accuracy_results.csv", index=False)
print("\n✅ Accuracy Report saved as: corrected_accuracy_results.csv")
print(df)
