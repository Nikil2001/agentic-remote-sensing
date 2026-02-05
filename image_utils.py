import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from unet_model import load_unet_model

# Load model once for utility functions (optional; in app.py you may load separately)
model = load_unet_model(num_classes=7)

# Preprocess image for model input
def preprocess_image(img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension

# Predict segmentation mask from image tensor
def predict_mask(model, tensor_image):
    with torch.no_grad():
        output = model(tensor_image)
        if output.shape[1] > 1:
            mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        else:
            mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    return mask

# Calculate class-wise land cover percentages
def calculate_land_cover_percentages(mask):
    total = mask.size
    classes, counts = np.unique(mask, return_counts=True)
    percentages = {
        class_names.get(int(cls), f"class_{cls}"): round((cnt / total) * 100, 2)
        for cls, cnt in zip(classes, counts)
    }
    return percentages

# Save RGB segmentation mask
def save_mask_image(mask, path):
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        rgb_mask[mask == class_id] = color
    Image.fromarray(rgb_mask).save(path)

# Constants
color_map = {
    0: (0, 255, 255),     # urban_land
    1: (255, 255, 0),     # agriculture_land
    2: (255, 0, 255),     # rangeland
    3: (0, 255, 0),       # forest_land
    4: (0, 0, 255),       # water
    5: (255, 255, 255),   # barren_land
    6: (0, 0, 0)          # unknown
}

class_names = {
    0: "urban_land",
    1: "agriculture_land",
    2: "rangeland",
    3: "forest_land",
    4: "water",
    5: "barren_land",
    6: "unknown"
}
