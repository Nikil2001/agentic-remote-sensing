import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from unet_model import UNet  # Must be in the same project

# ====== CONFIG ======
DATA_DIR = "data/train"
CLASS_CSV = "data/class_dict.csv"
IMG_SIZE = 128
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cpu")

# ====== STEP 1: Load color-class mapping ======
def load_class_map(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    color_to_index = {}
    for i, row in df.iterrows():
        rgb = (row['r'], row['g'], row['b'])
        color_to_index[rgb] = i
    return color_to_index

# ====== STEP 2: Custom Dataset Loader ======
class SatelliteDataset(Dataset):
    def __init__(self, image_paths, mask_paths, color_map):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.color_map = color_map
        self.img_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def mask_to_class(self, mask):
        mask = mask.resize((IMG_SIZE, IMG_SIZE))
        mask_np = np.array(mask)
        h, w, _ = mask_np.shape
        label_mask = np.zeros((h, w), dtype=np.int64)
        for color, index in self.color_map.items():
            match = np.all(mask_np == color, axis=-1)
            label_mask[match] = index
        return torch.from_numpy(label_mask)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("RGB")

        img_tensor = self.img_transform(img)
        mask_tensor = self.mask_to_class(mask)
        return img_tensor, mask_tensor

# ====== STEP 3: Get all image-mask pairs ======
def get_image_mask_pairs():
    files = os.listdir(DATA_DIR)
    images = sorted([f for f in files if f.endswith(".jpg")])
    masks = sorted([f for f in files if "mask" in f and f.endswith(".png")])

    image_paths = [os.path.join(DATA_DIR, f) for f in images]
    mask_paths = [os.path.join(DATA_DIR, f) for f in masks]
    return image_paths, mask_paths

# ====== STEP 4: Training Loop ======
def train():
    image_paths, mask_paths = get_image_mask_pairs()
    color_map = load_class_map(CLASS_CSV)

    # Train-val split
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42)

    train_set = SatelliteDataset(train_imgs, train_masks, color_map)
    val_set = SatelliteDataset(val_imgs, val_masks, color_map)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)

    model = UNet(num_classes=len(color_map)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("🚀 Training started...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, masks in tqdm(train_loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "unet_weights.pth")
    print("✅ Model saved as unet_weights.pth")

if __name__ == "__main__":
    train()
